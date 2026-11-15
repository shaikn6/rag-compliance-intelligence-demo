"""
llm_generator.py
Answer generator backed by a real small language model (Qwen2.5-1.5B-Instruct on CPU).

Same interface as MockGenerator: it receives the retrieved chunks, writes a short
answer from the best retrieved sentences only, and returns citations for them.
Every number in the answer is checked against those sentences; numbers that do not
appear there are reported so the reader can discount the answer.

On the free 2-vCPU hardware the model is slow, so it is loaded in the background,
generation has a time budget, and the passage-based answer is returned instead
when the model is not ready or does not finish in time.
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import List

from rag.generator import Citation, GeneratedAnswer, MockGenerator, _make_citation_marker, top_sentences
from rag.retriever import RetrievalResult

MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MAX_CONTEXT_SENTENCES = 5
MAX_NEW_TOKENS = 96
TIME_BUDGET_S = 60.0  # generation stops here and the passage-based answer is used
LOAD_WAIT_S = 300.0  # how long a request waits for a model that is still loading
_NUMBER = re.compile(r"\$?\d[\d,]*(?:\.\d+)?")

_SYSTEM = (
    "You answer compliance questions using only the numbered passages provided. "
    "Cite the passage you use by its number, like [1]. If the passages do not contain the answer, "
    "say that they do not. Keep the answer under four sentences."
)


def _numbers(text: str) -> set[str]:
    return {m.rstrip(".,").lstrip("$").replace(",", "") for m in _NUMBER.findall(text)}


def _flag_unverified(text: str, context: str, n_used: int) -> str:
    """Append a warning for any number in the answer that is not in the cited passages.
    Passage index markers [1]..[n_used] are not treated as claims."""
    unverified = sorted(_numbers(text) - _numbers(context) - {str(i) for i in range(1, n_used + 1)})
    if unverified:
        text += f"\n\n⚠️ Not found in the retrieved passages: {', '.join(unverified)}. Check the sources before relying on these."
    return text


def _cpu_quota() -> int:
    """CPUs this container may use. os.cpu_count() reports the host's cores, and
    running that many torch threads on a 2-vCPU quota makes generation crawl."""
    try:
        with open("/sys/fs/cgroup/cpu.max") as fh:
            quota, period = fh.read().split()
        if quota != "max":
            return max(1, int(quota) // int(period))
    except (OSError, ValueError):
        pass
    return os.cpu_count() or 1


class LLMGenerator:
    """Loads the model in a background thread so the app starts fast."""

    def __init__(self, model_id: str = MODEL_ID):
        self._model_id = model_id
        self._tok = None
        self._model = None
        self._loader: threading.Thread | None = None
        self._loaded = threading.Event()
        self._start_lock = threading.Lock()
        self._generate_lock = threading.Lock()
        self._fallback = MockGenerator()

    @property
    def ready(self) -> bool:
        return self._model is not None

    def start_loading(self) -> None:
        """Start downloading and loading the model (once)."""
        with self._start_lock:
            if self._loader is None:
                self._loader = threading.Thread(target=self._load, daemon=True)
                self._loader.start()

    def _load(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            t0 = time.perf_counter()
            torch.set_num_threads(_cpu_quota())
            tok = AutoTokenizer.from_pretrained(self._model_id)
            model = AutoModelForCausalLM.from_pretrained(self._model_id, dtype=torch.float32).eval()
            self._tok, self._model = tok, model
            print(f"{self._model_id} loaded in {time.perf_counter() - t0:.0f} s, {torch.get_num_threads()} threads")
        except Exception as exc:  # the passage-based answer is still served
            print(f"{self._model_id} failed to load: {exc!r}")
        finally:
            self._loaded.set()

    def _passage_answer(self, question: str, results: List[RetrievalResult], reason: str) -> GeneratedAnswer:
        fallback = self._fallback.generate(question, results)
        fallback.answer += f"\n\n_The language model {reason}, so this is the passage-based answer._"
        return fallback

    def generate(self, question: str, results: List[RetrievalResult]) -> GeneratedAnswer:
        if not results:
            return GeneratedAnswer(
                question=question,
                answer="No relevant compliance documents were found for your query.",
                citations=[],
                confidence=0.0,
            )
        import torch

        self.start_loading()
        self._loaded.wait(LOAD_WAIT_S)
        if not self.ready:
            reason = "failed to load" if self._loaded.is_set() else "is still loading"
            return self._passage_answer(question, results, reason)

        used = top_sentences(results, MAX_CONTEXT_SENTENCES)
        context = "\n".join(f"[{i}] ({r.source_doc}) {quote}" for i, (_, r, quote, _) in enumerate(used, 1))
        messages = [
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": f"Passages:\n{context}\n\nQuestion: {question}"},
        ]
        prompt = self._tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self._tok(prompt, return_tensors="pt", truncation=True, max_length=2048)
        with self._generate_lock, torch.no_grad():
            out = self._model.generate(
                **enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, max_time=TIME_BUDGET_S
            )
        new_tokens = out[0, enc["input_ids"].shape[1]:]
        eos = self._model.generation_config.eos_token_id
        eos = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        finished = bool(eos & set(new_tokens.tolist())) or len(new_tokens) >= MAX_NEW_TOKENS
        if not finished:
            return self._passage_answer(question, results, f"did not finish within {TIME_BUDGET_S:.0f} s")
        text = self._tok.decode(new_tokens, skip_special_tokens=True).strip()

        text = _flag_unverified(text, context, len(used))

        citations: List[Citation] = []
        for _, r, quote, section in used:
            marker = _make_citation_marker(r.source_doc, section)
            if all(c.marker != marker for c in citations):
                citations.append(
                    Citation(
                        marker=marker,
                        source_doc=r.source_doc,
                        section=section,
                        score=round(r.score, 4),
                        excerpt=quote[:120],
                    )
                )
        confidence = sum(sim for sim, _, _, _ in used) / len(used)  # retrieval similarity, not answer correctness
        return GeneratedAnswer(question=question, answer=text, citations=citations, confidence=round(confidence, 4))

"""Gradio demo — RAG Compliance Intelligence.

Natural-language Q&A across AML / GDPR / SOX / HIPAA / CCPA / PCI DSS compliance
documents with cited sources. FAISS + sentence-transformers + offline generator
(no API key).
"""
from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr

sys.path.insert(0, str(Path(__file__).parent))
from rag.llm_generator import LLMGenerator
from rag.pipeline import RAGPipeline

# Build the index once at startup.
PIPELINE = RAGPipeline()
DOCS = Path(__file__).parent / "data" / "sample_docs"
INDEXED = PIPELINE.index_directory(DOCS)
PIPELINE.query("What is the CTR threshold?")  # loads the answerability reranker before the first visitor
# Second pipeline whose answers are written by a small language model, loaded in
# the background from startup so the first request does not pay for the download.
LLM = LLMGenerator()
LLM.start_loading()
PIPELINE_LLM = RAGPipeline(generator=LLM)
PIPELINE_LLM.index_directory(DOCS)

SAMPLES = [
    "What is the SAR filing threshold?",
    "When must a data breach be reported to authorities?",
    "What are the GDPR rights of data subjects?",
    "What is the CTR reporting threshold?",
    "Explain SOX section 404 requirements",
    "What is Enhanced Due Diligence?",
    "How long must AML records be retained?",
    "Within how many days must individuals be notified of a HIPAA breach?",
    "How long does a business have to respond to a CCPA consumer request?",
    "What is the minimum password length under PCI DSS version 4.0?",
]


def answer(question: str, use_llm: bool):
    if not question or not question.strip():
        yield "Enter a question above.", "", ""
        return
    if use_llm and not LLM.ready:
        yield (
            "### Answer\n_Loading the language model (about 3 GB, downloaded after each restart). "
            "The answer appears here when it is ready; if it takes too long, the passage-based "
            "answer is shown instead._",
            "",
            "",
        )
    r = (PIPELINE_LLM if use_llm else PIPELINE).query(question)
    engine = "Qwen2.5-1.5B language model" if use_llm else "passage extraction"
    # Distinct source documents the answer is grounded in — the compliance regime(s).
    regimes = list(dict.fromkeys(c.source_doc for c in r.citations))
    grounded = f"  ·  **Grounded in:** {', '.join(regimes)}" if regimes else ""
    meta = (
        f"**Answer engine:** {engine}  ·  "
        f"**Retrieval similarity:** {r.confidence:.0%}  ·  "
        f"**Latency:** {r.latency_ms:.0f} ms  ·  "
        f"**Citations:** {len(r.citations)}{grounded}"
    )
    sources = ""
    for c in r.citations:
        sources += f"- **{c.marker}** — {c.excerpt[:180].strip()}…\n"
    scores = ""
    for res in (r.retrieval_results or [])[:5]:
        bar = "█" * max(1, round(res.score * 20))
        scores += f"`{res.score:.3f}` {bar}  {res.source_doc} §{res.section[:40]}\n"
    body = f"### Answer\n{r.answer}\n\n{meta}"
    if r.abstained:
        src = "### Sources\n_No answer was given, so no passage is cited._"
    elif sources:
        src = f"### 📎 Cited sources — the answer is grounded in these passages\n{sources}"
    else:
        src = ""
    sc = f"### Retrieval scores (semantic + keyword blend)\n{scores}" if scores else ""
    yield body, src, sc


with gr.Blocks(title="RAG Compliance Intelligence") as demo:
    gr.Markdown(
        "# ⚖️ RAG Compliance Intelligence\n"
        "Ask questions across **AML, GDPR, SOX, HIPAA, CCPA, and PCI DSS** compliance documents — "
        "answers come with cited sources. FAISS + sentence-transformers, no API key. By default a small local language model (Qwen2.5-1.5B) writes the answer grounded in the retrieved passages, with a graceful fall back to fast passage extraction if the model cannot load; the model can be wrong, so check the cited sources.  \n"
        f"*{INDEXED} document chunks indexed · "
        "source in the Files tab*"
    )
    with gr.Row():
        q = gr.Textbox(
            label="Your question",
            placeholder="e.g. What is the threshold for filing a Currency Transaction Report?",
            scale=4,
        )
        btn = gr.Button("Ask", variant="primary", scale=1)
    use_llm = gr.Checkbox(
        label="Write the answer with the language model (Qwen2.5-1.5B on the free CPU: about 15-17 seconds per answer when measured, and it falls back to the passage-based answer if the 3 GB model cannot load). "
        "Uncheck for the fast keyword-based answer assembled from the best passages.",
        value=True,
    )
    gr.Examples(SAMPLES, inputs=q, label="Sample questions")
    out = gr.Markdown()
    src = gr.Markdown()
    sc = gr.Markdown()
    # Several requests at once, so fast answers do not queue behind a slow model answer
    # (the model itself answers one request at a time).
    btn.click(answer, inputs=[q, use_llm], outputs=[out, src, sc], concurrency_limit=4)
    q.submit(answer, inputs=[q, use_llm], outputs=[out, src, sc], concurrency_limit=4)

    with gr.Accordion("How it works & measured evaluation", open=False):
        gr.Markdown(
            "**Retrieval stack:** sentence-transformers `all-MiniLM-L6-v2` embeddings in a FAISS "
            "`IndexFlatIP`, blended with BM25 keyword scores, then re-ranked by the "
            "`ms-marco-MiniLM-L-12-v2` cross-encoder.\n\n"
            "**Abstention:** a logistic answerability estimate must reach **0.92** or the system "
            "refuses rather than guess; questions naming terms the documents never use (e.g. FINRA) "
            "are refused outright.\n\n"
            "**Measured on the repo's own evaluation sets** (`tests/eval_set.py`, the frozen "
            "`answerability_*.json`, and `tests/evaluate.py`):\n"
            "| Metric | Result |\n|---|---|\n"
            "| KNOWN compliance questions answered | **14 / 14** (100% recall) |\n"
            "| Out-of-corpus / near-topic refused (frozen set 1) | **73 / 76** abstained |\n"
            "| Out-of-corpus / near-topic refused (frozen set 2) | **60 / 60** abstained |\n"
            "| Answerable wrongly refused (frozen set 1 / set 2) | **6 / 53  ·  12 / 57** |\n"
            "| Answerable wrongly refused across all 212 eval questions | **24 / 212 (11%)** |\n\n"
            "The **0.92** threshold is deliberately strict: raising it from an earlier 0.785 refuses "
            "near-topic look-alikes (EU AMLA, Singapore PDPA) that are out of corpus, at the honest "
            "cost of wrongly refusing more genuine questions — wrongly-refused rose from "
            "**13/212 (6%) to 24/212 (11%)**. We prefer a clean refusal over a confident wrong answer.\n\n"
            "**Limitations:** answers cover only the six bundled regimes (AML, GDPR, SOX, HIPAA, "
            "CCPA, PCI DSS); anything outside them is refused by design. The language model can still "
            "phrase things wrongly, so every number it writes is checked against the cited passages "
            "and flagged if unverified — always read the cited sources."
        )

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())

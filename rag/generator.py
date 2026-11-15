"""
generator.py
Mock LLM answer generator.

Composes a cited answer from the best retrieved sentences using
template-based synthesis — no external API key required.
The output mimics the format produced by a real LLM with citation markers
like "[aml_policy §SECTION 3]".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from rag.retriever import RetrievalResult


@dataclass
class Citation:
    """One source cited in the generated answer."""

    marker: str        # e.g. "[aml_policy §SECTION 3]"
    source_doc: str
    section: str
    score: float
    excerpt: str       # first 120 chars of the chunk


@dataclass
class GeneratedAnswer:
    """The complete output of the generator."""

    question: str
    answer: str
    citations: List[Citation] = field(default_factory=list)
    confidence: float = 0.0


def _make_citation_marker(source_doc: str, section: str) -> str:
    return f"[{source_doc} §{section}]"


def top_sentences(
    results: List[RetrievalResult], n: int
) -> List[tuple[float, RetrievalResult, str, str]]:
    """
    Best *n* distinct sentences across all retrieved passages, as
    (cosine similarity, passage, sentence, section of that sentence). Ranked by
    the sentence's hybrid score plus its passage's hybrid score, so passage
    context still counts.
    """
    ranked = sorted(
        (
            (score + r.score, sim, quote, section, r)
            for r in results
            for score, quote, sim, section in r.sentences
        ),
        key=lambda x: x[0],
        reverse=True,
    )
    picks: List[tuple[float, RetrievalResult, str, str]] = []
    seen: set[str] = set()
    for _, sim, quote, section, r in ranked:
        if quote not in seen:
            seen.add(quote)
            picks.append((sim, r, quote, section))
    return picks[:n]


class MockGenerator:
    """
    Extractive answer generator that requires no external API key.

    Strategy:
    1. Take the 3 best sentences the retriever scored (dense + BM25) across
       the retrieved passages.
    2. Compose a paragraph answer by stitching those sentences with
       transition phrasing and citation markers.
    3. Report confidence as the mean similarity of the grounding.
    """

    MAX_CONTEXT_CHUNKS = 3

    def generate(
        self,
        question: str,
        results: List[RetrievalResult],
    ) -> GeneratedAnswer:
        """
        Parameters
        ----------
        question : str
            The user's natural-language compliance question.
        results : List[RetrievalResult]
            Ordered (descending similarity) retrieval results.

        Returns
        -------
        GeneratedAnswer with answer text, citations, and confidence.
        """
        if not results:
            return GeneratedAnswer(
                question=question,
                answer=(
                    "No relevant compliance documents were found for your query. "
                    "Please ensure documents have been indexed."
                ),
                citations=[],
                confidence=0.0,
            )

        grounding = top_sentences(results, self.MAX_CONTEXT_CHUNKS)

        # Build citations
        citations: List[Citation] = []
        seen_markers: set[str] = set()
        for score, res, quote, section in grounding:
            marker = _make_citation_marker(res.source_doc, section)
            if marker in seen_markers:
                continue
            seen_markers.add(marker)
            citations.append(
                Citation(
                    marker=marker,
                    source_doc=res.source_doc,
                    section=section,
                    score=round(score, 4),
                    excerpt=quote[:120],
                )
            )

        # Compose the answer
        answer_parts: List[str] = []
        for i, (_, res, key_sentence, section) in enumerate(grounding):
            marker = _make_citation_marker(res.source_doc, section)
            if i == 0:
                answer_parts.append(f"{key_sentence} {marker}")
            elif i == 1:
                answer_parts.append(f"Additionally, {key_sentence.lower()} {marker}")
            else:
                answer_parts.append(f"Furthermore, {key_sentence.lower()} {marker}")

        answer_text = " ".join(answer_parts)
        if not answer_text.endswith("."):
            answer_text += "."

        confidence = float(
            sum(s for s, _, _, _ in grounding) / len(grounding)
        ) if grounding else 0.0

        return GeneratedAnswer(
            question=question,
            answer=answer_text,
            citations=citations,
            confidence=round(confidence, 4),
        )

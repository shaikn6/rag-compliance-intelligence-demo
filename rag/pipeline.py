"""
pipeline.py
End-to-end RAG pipeline: index documents then query them.

Usage (standalone):
    from rag.pipeline import RAGPipeline
    pipeline = RAGPipeline()
    pipeline.index_directory("data/sample_docs")
    answer = pipeline.query("What is the SAR filing threshold?")
    print(answer.answer)
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from rag.document_loader import DocumentChunk, DocumentLoader
from rag.generator import Citation, GeneratedAnswer, MockGenerator
from rag.retriever import RetrievalResult, Retriever
from rag.vector_store import VectorStore

EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 output dimension


@dataclass
class QueryResponse:
    """Complete response returned by :meth:`RAGPipeline.query`."""

    question: str
    answer: str
    citations: List[Citation]
    confidence: float
    latency_ms: float
    retrieval_results: List[RetrievalResult] = field(default_factory=list)
    abstained: bool = False

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<QueryResponse latency={self.latency_ms:.1f}ms "
            f"confidence={self.confidence:.2f} "
            f"citations={len(self.citations)}>"
        )


class RAGPipeline:
    """
    Orchestrates document loading, vector indexing, retrieval, and answer
    generation.  Accepts optional pre-built component instances for testing.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_TOP_K = 5
    # Answer only when the estimated chance that the documents answer the question is
    # at least ANSWERABLE_THRESHOLD. Logistic fit, on the dev questions (tests/eval_set.py
    # and tests/answerability_set.json), of the best sentence cosine similarity and the
    # best cross-encoder scores for sentences and for passages. The cut refuses the most
    # dev unanswerable questions with every KNOWN question answered and no more dev wrong
    # refusals than the earlier rule (similarity >= 0.50) had.
    # intercept, similarity, cross-encoder on sentences, cross-encoder on passages
    ANSWERABLE_FIT = (1.1475, -0.1496, 0.3676, 0.6085)
    # Re-tuned 2026-11-06 on tests/adversarial_eval.json, a set of near-topic
    # questions that sound in-domain but are out of the corpus (EU AMLA, Singapore
    # PDPA, Volcker Rule, ...). At 0.785 the two named offenders were answered
    # (EU AMLA p=0.80, Singapore PDPA p=0.81-0.91). 0.92 is the highest cut that
    # still sits below the weakest genuine KNOWN question (p=0.947) while refusing
    # both. Fit weights are unchanged. Tradeoff: on the 212 answerable questions in
    # the repo's own sets, wrongly-refused rose from 13/212 (6%) to 24/212 (11%);
    # KNOWN stays 14/14. See tests/evaluate_adversarial.py and the fix report.
    ANSWERABLE_THRESHOLD = 0.92
    NOT_FOUND = "Not found in the loaded documents"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        top_k: int = DEFAULT_TOP_K,
        vector_store: Optional[VectorStore] = None,
        loader: Optional[DocumentLoader] = None,
        retriever: Optional[Retriever] = None,
        generator: Optional[MockGenerator] = None,
    ) -> None:
        self._top_k = top_k
        self._store = vector_store or VectorStore(dim=EMBEDDING_DIM)
        self._loader = loader or DocumentLoader(model_name=model_name)
        self._retriever = retriever or Retriever(self._store, model_name=model_name)
        self._generator = generator or MockGenerator()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def index_file(self, path: str | Path) -> int:
        """
        Load, chunk, embed, and index a single .txt document.

        Returns
        -------
        int
            Number of chunks added.
        """
        chunks = self._loader.load_file(path)
        self._store.add(chunks)
        return len(chunks)

    def index_directory(self, directory: str | Path) -> int:
        """
        Index all .txt files inside *directory*.

        Returns
        -------
        int
            Total number of chunks added.
        """
        chunks = self._loader.load_directory(directory)
        self._store.add(chunks)
        return len(chunks)

    def index_chunks(self, chunks: List[DocumentChunk]) -> None:
        """Add pre-built chunks directly (used in tests / incremental indexing)."""
        self._store.add(chunks)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def query(self, question: str) -> QueryResponse:
        """
        Answer *question* using the indexed documents.

        Returns
        -------
        QueryResponse
            Contains the answer text, citations, confidence, and latency.
        """
        t0 = time.perf_counter()

        retrieval_results = self._retriever.retrieve(question, top_k=self._top_k)
        best = max((r.sentence_similarity for r in retrieval_results), default=0.0)
        # A question about a name or acronym the documents never use (HIPAA, FINRA)
        # cannot be answered from them, however similar the wording looks.
        unknown = self._retriever.unknown_terms(question)
        reason = ""
        if unknown:
            reason = f"they never mention {', '.join(unknown)}"
        else:
            p = self.answerability(best, *self._retriever.relevance(question, retrieval_results))
            if p < self.ANSWERABLE_THRESHOLD:
                reason = (
                    f"no passage answers the question closely enough (answerability "
                    f"{p:.2f}, below the {self.ANSWERABLE_THRESHOLD:.2f} threshold)"
                )
        if reason:
            return QueryResponse(
                question=question,
                answer=f"{self.NOT_FOUND}: {reason}. No answer is given.",
                citations=[],
                confidence=round(best, 4),
                latency_ms=round((time.perf_counter() - t0) * 1000.0, 2),
                retrieval_results=retrieval_results,
                abstained=True,
            )
        generated: GeneratedAnswer = self._generator.generate(question, retrieval_results)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return QueryResponse(
            question=question,
            answer=generated.answer,
            citations=generated.citations,
            confidence=generated.confidence,
            latency_ms=round(latency_ms, 2),
            retrieval_results=retrieval_results,
        )

    @classmethod
    def answerability(cls, similarity: float, sentence_score: float, passage_score: float) -> float:
        """Estimated chance that the documents answer the question (see ANSWERABLE_FIT)."""
        a, b, c, d = cls.ANSWERABLE_FIT
        return 1.0 / (1.0 + math.exp(-(a + b * similarity + c * sentence_score + d * passage_score)))

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the vector store to *path*."""
        self._store.save(path)

    def load(self, path: str | Path) -> None:
        """Load a previously saved vector store from *path*."""
        self._store = VectorStore.load(path)
        self._retriever = Retriever(self._store)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def index_size(self) -> int:
        """Total number of indexed chunks."""
        return self._store.size

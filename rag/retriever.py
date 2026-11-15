"""
retriever.py
Hybrid search over the FAISS vector store: dense cosine similarity blended with
BM25 keyword scores, at passage level and again at sentence level.
Returns ranked (chunk, score, source_doc, section) tuples plus the best sentence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder, SentenceTransformer

from rag.document_loader import DocumentChunk
from rag.vector_store import VectorStore

# Weight of the BM25 score in the blend (dense gets 1 - LEXICAL_WEIGHT).
LEXICAL_WEIGHT = 0.5
SENTENCES_PER_CHUNK = 3
# Reads question and sentence together, so it can tell a sentence that answers the
# question from one that only shares its words (33M parameters, fast on CPU).
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-12-v2"


@lru_cache(maxsize=1)
def _reranker() -> CrossEncoder:
    """One shared copy: the app runs two pipelines over the same documents."""
    return CrossEncoder(RERANK_MODEL, max_length=512)

_STOPWORDS = set(
    "a an and are as at be by can do does for from has have how i in is it its many "
    "much must of on or should that the their them they this to under was what when "
    "where which who whom why will with within".split()
)
# A sentence ends after a word or ")" (not after list numbers like "1.").
_SENTENCE_END = re.compile(r"(?<=[A-Za-z\)][.!?])\s+")
# Heading lines ("SECTION 3: ...", "3.1 Right of Access (Article 15)") label the
# sentences under them but are not answer candidates.
_HEADING = re.compile(r"^(SECTION \d+:|\d+\.\d+ )[^.]*$")


def tokenize(text: str) -> List[str]:
    """Lowercase word tokens without stopwords; a trailing plural 's' is dropped."""
    tokens = []
    for t in re.findall(r"[a-z0-9]+", text.lower()):
        if t in _STOPWORDS:
            continue
        if len(t) > 3 and t.endswith("s") and not t.endswith("ss"):
            t = t[:-1]
        tokens.append(t)
    return tokens


def split_sentences(text: str, section: str = "") -> List[Tuple[str, str, str]]:
    """
    Split a passage into (section, nearest heading, sentence) triples; headings
    are not sentences. *section* is the SECTION in force where the passage starts.
    """
    triples: List[Tuple[str, str, str]] = []
    heading, body = "", []
    for line in text.splitlines() + [""]:
        line = line.strip()
        if line and not _HEADING.match(line):
            body.append(line)
            continue
        triples += [(section, heading, x) for x in _SENTENCE_END.split(" ".join(body)) if x.strip()]
        body = []
        if line:
            heading = line
            if line.startswith("SECTION "):
                section = line
    return triples


def named_terms(question: str) -> List[str]:
    """Words written with capitals (acronyms, names), except a capitalised first word.
    In a Title Case question capitals mean nothing, so only acronyms count there."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'-]*", question)
    title_case = sum(w[0].isupper() for w in words) > len(words) / 2
    return [
        w
        for i, w in enumerate(words)
        if sum(c.isupper() for c in w) > (1 if title_case or not i else 0)
    ]


def _blend(dense: np.ndarray, lexical: np.ndarray) -> np.ndarray:
    """Convex blend of cosine similarity and max-normalised BM25."""
    top = lexical.max() if lexical.size else 0.0
    lex = lexical / top if top > 0 else np.zeros_like(lexical)
    return (1.0 - LEXICAL_WEIGHT) * dense + LEXICAL_WEIGHT * lex


@dataclass
class RetrievalResult:
    """A single retrieved passage with its similarity score and provenance."""

    chunk: DocumentChunk
    score: float  # hybrid score used for ranking
    source_doc: str
    section: str
    dense_score: float = 0.0  # cosine similarity of the passage to the query
    # (hybrid score, sentence with its heading, cosine similarity, section), best first
    sentences: List[Tuple[float, str, float, str]] = field(default_factory=list)
    sentence_similarity: float = 0.0  # best sentence cosine similarity in this passage

    def __repr__(self) -> str:  # pragma: no cover
        preview = self.chunk.text[:60].replace("\n", " ")
        return (
            f"<Result score={self.score:.3f} "
            f"doc={self.source_doc!r} "
            f"section={self.section!r} "
            f"text={preview!r}>"
        )


class Retriever:
    """
    Encodes a natural-language query and retrieves the top-k chunks by a blend
    of semantic similarity and BM25 keyword overlap, then picks the best
    sentence inside each returned chunk the same way.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(
        self,
        vector_store: VectorStore,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self._store = vector_store
        self._model_name = model_name
        self._model = None
        self._indexed = -1  # store size the keyword/sentence index was built for

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        return self._model.encode(
            texts, normalize_embeddings=True, show_progress_bar=False, batch_size=64
        )

    def _build(self) -> None:
        """(Re)build the BM25 and sentence indexes when the store has changed."""
        chunks = self._store.chunks
        self._bm25 = BM25Okapi([tokenize(c.text) or [""] for c in chunks])
        self._sent_text, self._sent_heading, self._sent_section, owner = [], [], [], []
        for i, c in enumerate(chunks):
            start = c.start_section or c.section
            triples = split_sentences(c.text, start) or [(start, "", c.text)]
            self._sent_section += [sec for sec, _, _ in triples]
            self._sent_heading += [h for _, h, _ in triples]
            self._sent_text += [" ".join(t.split()) for _, _, t in triples]
            owner += [i] * len(triples)
        self._vocab = {t for c in chunks for t in tokenize(c.text)}
        self._sent_owner = np.array(owner)
        self._sent_vecs = self._encode(self._sent_text)
        self._sent_bm25 = BM25Okapi([tokenize(t) or [""] for t in self._sent_text])
        self._pos = {id(c): i for i, c in enumerate(chunks)}
        self._indexed = self._store.size

    def _quote(self, i: int) -> str:
        """Sentence prefixed with its heading, e.g. '3.3 Right to Erasure (Article 17): ...'."""
        h = self._sent_heading[i]
        return f"{h}: {self._sent_text[i]}" if h else self._sent_text[i]

    def unknown_terms(self, question: str) -> List[str]:
        """Names and acronyms in *question* that never occur in the indexed documents."""
        if self._store.size == 0:
            return []
        if self._indexed != self._store.size:
            self._build()
        return [
            w
            for w in named_terms(question)
            if any(len(t) > 1 and t not in self._vocab for t in tokenize(w))
        ]

    def relevance(self, question: str, results: List[RetrievalResult]) -> Tuple[float, float]:
        """Best cross-encoder score of *question* against the quoted sentences of *results*,
        and against their whole passages."""
        sentences = [quote for r in results for _, quote, _, _ in r.sentences]
        if not sentences:
            return float("-inf"), float("-inf")
        pairs = [(question, s) for s in sentences] + [(question, r.chunk.text) for r in results]
        scores = _reranker().predict(pairs, show_progress_bar=False)
        return float(scores[: len(sentences)].max()), float(scores[len(sentences):].max())

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self, query: str, top_k: int = 5
    ) -> List[RetrievalResult]:
        """
        Retrieve the top-k chunks most relevant to *query*.

        Parameters
        ----------
        query : str
            The natural-language question or search string.
        top_k : int
            Maximum number of results to return.

        Returns
        -------
        List[RetrievalResult] sorted descending by hybrid score.
        """
        n = self._store.size
        if n == 0:
            return []
        if self._indexed != n:
            self._build()

        query_vec: np.ndarray = self._encode([query])[0]
        query_tokens = tokenize(query)

        # Score every chunk: the demo corpus is small, so no candidate cut-off.
        dense = np.zeros(n, dtype=np.float32)
        for chunk, score in self._store.search(query_vec, top_k=n):
            dense[self._pos[id(chunk)]] = score
        hybrid = _blend(dense, np.asarray(self._bm25.get_scores(query_tokens)))

        top = np.argsort(-hybrid)[:top_k]

        # Sentence level: the same blend over every sentence of the top chunks,
        # so the answer can quote the one sentence that states the fact.
        cand = np.flatnonzero(np.isin(self._sent_owner, top))
        sent_dense = self._sent_vecs[cand] @ query_vec
        sent_score = _blend(sent_dense, self._sent_bm25.get_scores(query_tokens)[cand])

        results: List[RetrievalResult] = []
        for idx in top:
            own = np.flatnonzero(self._sent_owner[cand] == idx)
            own = own[np.argsort(-sent_score[own])][:SENTENCES_PER_CHUNK]
            chunk = self._store.chunks[idx]
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(hybrid[idx]),
                    source_doc=chunk.source_doc,
                    section=chunk.section,
                    dense_score=float(dense[idx]),
                    sentences=[
                        (
                            float(sent_score[j]),
                            self._quote(cand[j]),
                            float(sent_dense[j]),
                            self._sent_section[cand[j]] or chunk.section,
                        )
                        for j in own
                    ],
                    sentence_similarity=float(sent_dense[own].max()),
                )
            )

        return results

"""
vector_store.py
FAISS IndexFlatIP wrapper for approximate nearest-neighbour search.
Embeddings are L2-normalised upstream so inner-product == cosine similarity.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import pickle
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from rag.document_loader import DocumentChunk

# Secret used to sign/verify the chunks pickle file. There is no usable default:
# a baked-in key is public in source, so an attacker could forge a signature over a
# malicious pickle and achieve code execution on load(). Require an explicit env key.
_HMAC_KEY_RAW = os.environ.get("VECTOR_STORE_HMAC_KEY")
_HMAC_KEY: bytes | None = _HMAC_KEY_RAW.encode() if _HMAC_KEY_RAW else None


def _require_hmac_key() -> bytes:
    if not _HMAC_KEY:
        raise RuntimeError(
            "VECTOR_STORE_HMAC_KEY is not set. Refusing to sign or load a pickled store "
            "without an explicit secret key, because a default key would let an attacker "
            "forge the signature and execute code on load()."
        )
    return _HMAC_KEY


class VectorStore:
    """
    Thin wrapper around a FAISS IndexFlatIP index.

    Because embeddings are normalised (unit vectors), inner product
    between two vectors equals their cosine similarity, ranging [−1, 1].
    Higher is more similar.
    """

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(dim)
        self._chunks: List[DocumentChunk] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Number of vectors currently stored."""
        return self._index.ntotal

    @property
    def chunks(self) -> List[DocumentChunk]:
        """Indexed chunks, in index order."""
        return self._chunks

    def add(self, chunks: List[DocumentChunk]) -> None:
        """Add *chunks* to the index (embeddings must already be populated)."""
        if not chunks:
            return
        matrix = np.stack([c.embedding for c in chunks]).astype(np.float32)
        self._index.add(matrix)
        self._chunks.extend(chunks)

    def search(
        self, query_vector: np.ndarray, top_k: int = 5
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Return the top-k most similar chunks along with their similarity scores.

        Parameters
        ----------
        query_vector : np.ndarray, shape (dim,)
            L2-normalised query embedding.
        top_k : int
            Number of results to return.

        Returns
        -------
        list of (DocumentChunk, score) sorted descending by score.
        """
        if self.size == 0:
            return []

        effective_k = min(top_k, self.size)
        query = query_vector.astype(np.float32).reshape(1, -1)
        scores, indices = self._index.search(query, effective_k)

        results: List[Tuple[DocumentChunk, float]] = []
        for idx, score in zip(indices[0], scores[0]):
            if idx < 0:
                continue
            results.append((self._chunks[idx], float(score)))

        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist the FAISS index and chunk list to *path* directory."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path / "index.faiss"))
        blob = pickle.dumps(self._chunks)
        sig = hmac.new(_require_hmac_key(), blob, hashlib.sha256).hexdigest()
        with open(path / "chunks.pkl", "wb") as fh:
            fh.write(blob)
        with open(path / "chunks.pkl.sig", "w") as fh:
            fh.write(sig)

    @classmethod
    def load(cls, path: str | Path) -> "VectorStore":
        """Load a previously saved store from *path* directory.

        Verifies the HMAC signature written by ``save`` before deserialising.
        """
        path = Path(path)
        index: faiss.IndexFlatIP = faiss.read_index(str(path / "index.faiss"))

        sig_path = path / "chunks.pkl.sig"
        if not sig_path.exists():
            raise ValueError(
                f"Integrity signature missing at {sig_path}. "
                "Re-index documents to regenerate a signed store."
            )

        with open(path / "chunks.pkl", "rb") as fh:
            blob = fh.read()

        with open(sig_path) as fh:
            expected_sig = fh.read().strip()

        actual_sig = hmac.new(_require_hmac_key(), blob, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(actual_sig, expected_sig):
            raise ValueError(
                "Vector store integrity check failed: chunks.pkl signature mismatch. "
                "The file may have been tampered with."
            )

        chunks: List[DocumentChunk] = pickle.loads(blob)  # noqa: S301 — sig verified above

        dim = index.d
        store = cls(dim=dim)
        store._index = index
        store._chunks = chunks
        return store

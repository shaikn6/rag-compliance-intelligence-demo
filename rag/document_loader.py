"""
document_loader.py
Chunk compliance documents into overlapping segments and compute
sentence-transformer embeddings for each chunk.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


@dataclass
class DocumentChunk:
    """A single text chunk with its embedding and provenance metadata."""

    chunk_id: str
    text: str
    source_doc: str
    section: str
    char_start: int
    char_end: int
    embedding: np.ndarray = field(default_factory=lambda: np.array([]))
    start_section: str = ""  # section heading in force where the chunk begins

    def __repr__(self) -> str:  # pragma: no cover
        preview = self.text[:60].replace("\n", " ")
        return f"<Chunk {self.chunk_id} | {self.source_doc} | {preview!r}>"


class DocumentLoader:
    """
    Loads plain-text compliance documents, splits them into
    ~200-word chunks with configurable overlap, and embeds each
    chunk using a sentence-transformer model.
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    CHUNK_SIZE_WORDS = 200
    OVERLAP_WORDS = 40

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        chunk_size: int = CHUNK_SIZE_WORDS,
        overlap: int = OVERLAP_WORDS,
    ) -> None:
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._model_name = model_name
        self._model = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_file(self, path: str | Path) -> list[DocumentChunk]:
        """Load a single .txt file and return a list of embedded chunks."""
        path = Path(path)
        raw = path.read_text(encoding="utf-8")
        doc_name = path.stem
        chunks = self._split(raw, doc_name)
        return self._embed(chunks)

    def load_directory(self, directory: str | Path) -> list[DocumentChunk]:
        """Load every .txt file inside *directory* (non-recursive)."""
        directory = Path(directory)
        all_chunks: list[DocumentChunk] = []
        for txt_file in sorted(directory.glob("*.txt")):
            all_chunks.extend(self.load_file(txt_file))
        return all_chunks

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_section(self, text: str) -> str:
        """
        Heuristically detect the nearest section heading above the chunk.
        Returns the heading text or 'General' if none found.
        """
        pattern = re.compile(
            r"^(SECTION\s+\d+[\.\d]*[:\s]+[A-Z][A-Za-z0-9 \(\)\/\-]+)$",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(text))
        if matches:
            return matches[-1].group(1).strip()
        return "General"

    def _split(self, text: str, doc_name: str) -> list[DocumentChunk]:
        """
        Split *text* into overlapping word-based windows.
        Each chunk is labelled with the last visible section heading.
        """
        words = list(re.finditer(r"\S+", text))
        chunks: list[DocumentChunk] = []
        step = self._chunk_size - self._overlap
        chunk_index = 0

        for start_word in range(0, len(words), step):
            end_word = start_word + self._chunk_size
            chunk_words = words[start_word:end_word]
            if not chunk_words:
                break

            # Slice the original text so line breaks (headings) are kept.
            char_start = chunk_words[0].start()
            char_end = chunk_words[-1].end()
            chunk_text = text[char_start:char_end]

            # Detect section from the original text up to this chunk's end,
            # preserving newlines so the regex anchors in _detect_section work.
            section = self._detect_section(text[:char_end])
            first_line_end = text.find("\n", char_start)
            start_section = self._detect_section(
                text[: first_line_end if first_line_end >= 0 else char_end]
            )

            chunk_id = f"{doc_name}_{chunk_index:04d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    text=chunk_text,
                    source_doc=doc_name,
                    section=section,
                    char_start=char_start,
                    char_end=char_end,
                    start_section=start_section,
                )
            )
            chunk_index += 1

            if end_word >= len(words):
                break

        return chunks

    def _embed(self, chunks: list[DocumentChunk]) -> list[DocumentChunk]:
        """Compute L2-normalised embeddings for every chunk in *chunks*."""
        if not chunks:
            return chunks

        texts = [c.text for c in chunks]
        if self._model is None:
            self._model = SentenceTransformer(self._model_name)
        embeddings: np.ndarray = self._model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32,
        )
        for chunk, emb in zip(chunks, embeddings):
            chunk.embedding = emb
        return chunks

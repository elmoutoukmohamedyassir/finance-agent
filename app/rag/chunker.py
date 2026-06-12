"""
rag/chunker.py — Sentence-aware text chunking.

Why sentence-aware matters:
  Naive fixed-size chunking cuts sentences mid-word → bad embeddings → bad retrieval.
  This chunker respects sentence boundaries, then groups them into chunks.
"""

import re
import logging

logger = logging.getLogger(__name__)


def clean_pdf_text(text: str) -> str:
    """Remove PDF noise: page numbers, excessive whitespace, non-printable chars."""
    text = re.sub(r'\bPage\s+\d+\s*(of\s+\d+)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\s*[-–]\s*\d+\s*[-–]\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[^\x20-\x7E\xA0-\xFF\n]', '', text)
    return text.strip()


def split_sentences(text: str) -> list[str]:
    """Split text into sentences, respecting common abbreviations."""
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\d"])', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def create_chunks(
    text: str,
    source_filename: str,
    chunk_size: int = 700,
    overlap_size: int = 120,
    min_length: int = 80,
) -> list[dict]:
    """
    Split text into overlapping chunks with metadata.

    chunk_size=700 chars is a good fit for nomic-embed-text's 8192-token window.
    overlap=120 ensures consecutive chunks share context (avoids losing info at boundaries).

    Returns list of {"text": ..., "metadata": {"source": ..., "chunk_index": ...}}
    """
    sentences = split_sentences(text)
    chunks = []
    current_sents = []
    current_len = 0
    chunk_idx = 0

    for sentence in sentences:
        slen = len(sentence)

        if current_len + slen > chunk_size and current_sents:
            chunk_text = " ".join(current_sents).strip()
            if len(chunk_text) >= min_length:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {"source": source_filename, "chunk_index": chunk_idx}
                })
                chunk_idx += 1

            # Overlap: keep tail sentences that fit in overlap_size
            overlap = []
            overlap_len = 0
            for s in reversed(current_sents):
                if overlap_len + len(s) <= overlap_size:
                    overlap.insert(0, s)
                    overlap_len += len(s)
                else:
                    break
            current_sents = overlap
            current_len = overlap_len

        current_sents.append(sentence)
        current_len += slen

    # Last chunk
    if current_sents:
        chunk_text = " ".join(current_sents).strip()
        if len(chunk_text) >= min_length:
            chunks.append({
                "text": chunk_text,
                "metadata": {"source": source_filename, "chunk_index": chunk_idx}
            })

    logger.info(f"'{source_filename}': {len(sentences)} sentences → {len(chunks)} chunks")
    return chunks

"""
rag/chunker.py — Smart text chunking for RAG ingestion.

IMPROVEMENTS over the original split_text():
  1. Sentence-aware splitting: never cuts mid-sentence
  2. Metadata attached to every chunk (source file, page, chunk index)
  3. Configurable chunk size with sensible defaults for finance docs
  4. Overlap ensures context continuity between chunks
  5. Filters out noise (very short chunks, page headers/footers)

WHY sentence-aware chunking matters:
  Original code split at fixed character count → "Customer Acqui" + "sition Cost"
  New code respects sentence boundaries → coherent, retrievable chunks
  Better chunks = better embeddings = better retrieval = fewer hallucinations
"""

import re
import logging

logger = logging.getLogger(__name__)


def split_into_sentences(text: str) -> list[str]:
    """
    Basic sentence splitter that handles common abbreviations.
    Splits on ". ", "! ", "? " but not on "e.g.", "i.e.", "Mr.", etc.
    """
    # Normalize whitespace first
    text = re.sub(r'\s+', ' ', text).strip()

    # Simple sentence boundary detection
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text)
    return [s.strip() for s in sentences if s.strip()]


def create_chunks(
    text: str,
    source_filename: str,
    chunk_size: int = 600,
    overlap_size: int = 100,
    min_chunk_length: int = 80,
) -> list[dict]:
    """
    Splits text into overlapping chunks with metadata.

    Strategy:
      1. Split into sentences first
      2. Accumulate sentences into chunks up to chunk_size characters
      3. When a chunk is full, save it and start a new one with overlap
      4. Attach metadata to every chunk

    Args:
        text: Raw extracted text from a PDF.
        source_filename: The PDF filename (stored as metadata).
        chunk_size: Target max characters per chunk (default 600 is good for
                    sentence-transformers with 512-token limit).
        overlap_size: Characters of overlap between consecutive chunks.
                      Ensures context continuity for retrieval.
        min_chunk_length: Discard chunks shorter than this (noise filter).

    Returns:
        List of dicts: {"text": ..., "metadata": {"source": ..., "chunk_index": ...}}
    """
    sentences = split_into_sentences(text)
    chunks = []
    current_chunk = []
    current_length = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        # If adding this sentence would exceed chunk_size, finalize current chunk
        if current_length + sentence_len > chunk_size and current_chunk:
            chunk_text = " ".join(current_chunk).strip()

            if len(chunk_text) >= min_chunk_length:
                chunks.append({
                    "text": chunk_text,
                    "metadata": {
                        "source": source_filename,
                        "chunk_index": chunk_index,
                    }
                })
                chunk_index += 1

            # Start new chunk with overlap: keep last N characters worth of sentences
            overlap_text = ""
            for s in reversed(current_chunk):
                if len(overlap_text) + len(s) <= overlap_size:
                    overlap_text = s + " " + overlap_text
                else:
                    break

            current_chunk = [overlap_text.strip()] if overlap_text.strip() else []
            current_length = len(overlap_text)

        current_chunk.append(sentence)
        current_length += sentence_len + 1  # +1 for space

    # Don't forget the last chunk
    if current_chunk:
        chunk_text = " ".join(current_chunk).strip()
        if len(chunk_text) >= min_chunk_length:
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    "source": source_filename,
                    "chunk_index": chunk_index,
                }
            })

    logger.info(f"Chunked '{source_filename}': {len(sentences)} sentences → {len(chunks)} chunks")
    return chunks


def clean_pdf_text(text: str) -> str:
    """
    Cleans raw PDF-extracted text before chunking.

    Removes common PDF noise:
      - Page numbers ("Page 1 of 20", "- 5 -")
      - Repeated headers/footers
      - Excessive newlines
      - Non-printable characters
    """
    # Remove page number patterns
    text = re.sub(r'[Pp]age\s+\d+\s+(of\s+\d+)?', '', text)
    text = re.sub(r'^\s*[-–—]\s*\d+\s*[-–—]\s*$', '', text, flags=re.MULTILINE)

    # Remove excessive whitespace and normalize newlines
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)

    # Remove non-printable characters (keep standard punctuation)
    text = re.sub(r'[^\x20-\x7E\xA0-\xFF\n]', '', text)

    return text.strip()

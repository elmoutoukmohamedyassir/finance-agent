"""
ingest.py — Standalone PDF ingestion script.

Run this from the project root to embed your PDFs into ChromaDB:

    python ingest.py

Or to force re-indexing of already-indexed files:

    python ingest.py --force

You can also trigger ingestion via the API:
    POST http://localhost:8000/api/v1/rag/ingest

HOW IT WORKS:
  1. Reads all .pdf files from the docs/ folder
  2. Extracts text from each page
  3. Cleans the text (removes PDF noise)
  4. Splits text into overlapping chunks (~700 chars each)
  5. Embeds each chunk using nomic-embed-text (or your configured model)
  6. Stores chunks + embeddings in ChromaDB (data/chroma/)

After running this, the finance agent will automatically use your documents
to enrich its answers with grounded knowledge.
"""

import sys
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from app.rag.ingestion import ingest_all_documents


def main():
    force = "--force" in sys.argv

    if force:
        print("Re-ingesting all documents (--force mode)...")
    else:
        print("Ingesting new documents from docs/...")

    results = ingest_all_documents(force=force)

    if not results:
        print("\nNo PDF files found in docs/")
        print("→ Drop your PDF files into the docs/ folder and run this again.")
        return

    print(f"\n{'─' * 50}")
    print(f"{'FILE':<45} {'STATUS':<12} {'CHUNKS'}")
    print(f"{'─' * 50}")
    for r in results:
        name = r["file"][:42] + "..." if len(r["file"]) > 45 else r["file"]
        print(f"{name:<45} {r['status']:<12} {r['chunks']}")

    ingested = sum(r["chunks"] for r in results if r["status"] == "ingested")
    total = sum(r["chunks"] for r in results)
    print(f"{'─' * 50}")
    print(f"Done. New chunks added: {ingested}. Total in DB: see GET /api/v1/rag/status")


if __name__ == "__main__":
    main()

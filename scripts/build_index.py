"""
Builds the FAISS vector index from ingested IEEE RAS documents.
Chunks text, computes normalized sentence embeddings, and saves index.faiss + metadata.json.
"""

import json
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any
import faiss

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import (
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    VECTORSTORE_DIR,
    FAISS_INDEX_PATH,
    METADATA_PATH,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_DIM
)
from src.embeddings import get_embedding_manager
from src.utils import recursive_chunk_text

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_faiss_index():
    """Chunk raw documents, generate embeddings, and build FAISS vector index."""
    raw_file = RAW_DATA_DIR / "ieee_ras_raw_docs.json"
    if not raw_file.exists():
        logger.error("Raw documents file not found at %s. Run scrape_data.py first.", raw_file)
        sys.exit(1)

    with open(raw_file, "r", encoding="utf-8") as f:
        documents: List[Dict[str, Any]] = json.load(f)

    logger.info("Loaded %d raw documents from %s", len(documents), raw_file)

    # Prepare chunked records
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    VECTORSTORE_DIR.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, Any]] = []
    chunk_texts: List[str] = []
    chunk_counter = 0

    for doc_idx, doc in enumerate(documents):
        doc_url = doc.get("url", "https://www.ieee-ras.org/")
        doc_title = doc.get("title", "IEEE RAS Resource")
        doc_category = doc.get("category", "General")
        doc_text = doc.get("text", "")

        raw_chunks = recursive_chunk_text(doc_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)

        for chunk_str in raw_chunks:
            chunk_metadata = {
                "chunk_id": chunk_counter,
                "doc_id": doc_idx,
                "title": doc_title,
                "url": doc_url,
                "category": doc_category,
                "text": chunk_str
            }
            all_chunks.append(chunk_metadata)
            chunk_texts.append(chunk_str)
            chunk_counter += 1

    logger.info("Generated %d chunks from %d documents.", len(all_chunks), len(documents))

    # Save processed chunks JSON
    processed_file = PROCESSED_DATA_DIR / "ieee_ras_chunks.json"
    with open(processed_file, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)
    logger.info("Saved processed chunks to %s", processed_file)

    # Generate vector embeddings
    logger.info("Generating dense embeddings for %d chunks using SentenceTransformers...", len(chunk_texts))
    embedding_mgr = get_embedding_manager()
    embeddings = embedding_mgr.encode_texts(chunk_texts)

    logger.info("Embeddings matrix generated with shape: %s", embeddings.shape)

    # Initialize FAISS IndexFlatIP (Inner Product = Cosine Similarity with normalized vectors)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)

    logger.info("Added %d vectors to FAISS index. Total in index: %d", embeddings.shape[0], index.ntotal)

    # Save FAISS index and metadata
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, indent=2, ensure_ascii=False)

    logger.info("FAISS index saved to %s", FAISS_INDEX_PATH)
    logger.info("Metadata saved to %s", METADATA_PATH)
    print(f"SUCCESS: Vector index built with {index.ntotal} vectors.")


if __name__ == "__main__":
    build_faiss_index()

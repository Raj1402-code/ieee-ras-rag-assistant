"""
FAISS vector retriever for semantic search and context retrieval.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import faiss
import numpy as np

from src.config import (
    FAISS_INDEX_PATH,
    METADATA_PATH,
    TOP_K_RETRIEVAL,
    SIMILARITY_THRESHOLD
)
from src.embeddings import get_embedding_manager

logger = logging.getLogger(__name__)


class Retriever:
    """FAISS-based vector retriever for IEEE RAS documents."""

    def __init__(self):
        self.index: Optional[faiss.Index] = None
        self.metadata: List[Dict[str, Any]] = []
        self.embedding_mgr = get_embedding_manager()
        self.load_index()

    def load_index(self) -> bool:
        """Load the FAISS index and corresponding metadata file."""
        if not FAISS_INDEX_PATH.exists() or not METADATA_PATH.exists():
            logger.warning("FAISS index or metadata does not exist at %s", FAISS_INDEX_PATH)
            return False

        try:
            self.index = faiss.read_index(str(FAISS_INDEX_PATH))
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)
            return True
        except Exception as e:
            logger.error("Failed to load FAISS index: %s", e)
            return False

    def is_ready(self) -> bool:
        """Check if retriever has a loaded index and metadata."""
        return self.index is not None and len(self.metadata) > 0

    def retrieve(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        threshold: float = SIMILARITY_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Perform vector similarity search against the FAISS index.
        Returns top matching chunks, similarity scores, and relevance judgment.
        """
        if not self.is_ready():
            # Try reloading if not ready
            if not self.load_index():
                return {
                    "chunks": [],
                    "is_relevant": False,
                    "max_score": 0.0,
                    "query": query,
                    "error": "Vector index not found or uninitialized."
                }

        # Encode query to normalized 2D vector
        query_vec = self.embedding_mgr.encode_query(query)

        # Ensure top_k does not exceed total vectors
        k = min(top_k, self.index.ntotal)
        if k <= 0:
            return {
                "chunks": [],
                "is_relevant": False,
                "max_score": 0.0,
                "query": query
            }

        # Search index (IndexFlatIP returns cosine similarity directly)
        scores, indices = self.index.search(query_vec, k)
        
        chunks = []
        max_score = 0.0

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            
            score_val = float(score)
            if score_val > max_score:
                max_score = score_val

            meta = self.metadata[idx]
            chunks.append({
                "chunk_id": int(idx),
                "similarity_score": round(score_val, 4),
                "text": meta.get("text", ""),
                "title": meta.get("title", "IEEE RAS Resource"),
                "url": meta.get("url", "https://www.ieee-ras.org/"),
                "category": meta.get("category", "General")
            })

        # Determine if query is sufficiently relevant to IEEE RAS context
        is_relevant = max_score >= threshold and len(chunks) > 0

        return {
            "chunks": chunks,
            "is_relevant": is_relevant,
            "max_score": round(max_score, 4),
            "query": query
        }


_retriever_instance = None


def get_retriever() -> Retriever:
    """Singleton getter for the FAISS retriever."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = Retriever()
    return _retriever_instance

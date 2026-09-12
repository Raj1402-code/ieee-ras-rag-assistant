"""
Embedding model manager for generating dense semantic vector representations.
"""

from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from src.config import EMBEDDING_MODEL_NAME


class EmbeddingManager:
    """Singleton wrapper for SentenceTransformer embedding operations."""
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingManager, cls).__new__(cls)
            cls._model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        return cls._instance

    @property
    def model(self) -> SentenceTransformer:
        return self._model

    def encode_texts(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of text strings into normalized float32 embeddings.
        Normalizing enables exact cosine similarity using FAISS IndexFlatIP.
        """
        embeddings = self._model.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        return embeddings.astype(np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a normalized 2D vector (1, d)."""
        emb = self.encode_texts([query])
        return emb


_embedding_manager_instance = None

def get_embedding_manager() -> EmbeddingManager:
    """Retrieve the shared singleton EmbeddingManager."""
    global _embedding_manager_instance
    if _embedding_manager_instance is None:
        _embedding_manager_instance = EmbeddingManager()
    return _embedding_manager_instance

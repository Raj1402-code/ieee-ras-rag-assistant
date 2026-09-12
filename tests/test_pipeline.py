"""
Automated unit and integration tests for the IEEE RAS RAG pipeline.
"""

import unittest
import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.config import FAISS_INDEX_PATH, METADATA_PATH, EMBEDDING_DIM
from src.embeddings import get_embedding_manager
from src.retriever import get_retriever
from src.rag import get_rag_pipeline


class TestIEEERASRAG(unittest.TestCase):
    """Test suite covering embedding, FAISS retrieval, and RAG guardrails."""

    def setUp(self):
        self.embedding_mgr = get_embedding_manager()
        self.retriever = get_retriever()
        self.rag_pipeline = get_rag_pipeline()

    def test_01_index_and_metadata_exist(self):
        """Verify vector index and metadata files were generated."""
        self.assertTrue(FAISS_INDEX_PATH.exists(), "FAISS index file is missing.")
        self.assertTrue(METADATA_PATH.exists(), "Metadata JSON file is missing.")
        self.assertTrue(self.retriever.is_ready(), "Retriever failed to initialize index.")
        self.assertGreater(self.retriever.index.ntotal, 0, "Vector index is empty.")

    def test_02_embedding_generation(self):
        """Verify SentenceTransformer generates normalized embeddings of correct dimension."""
        text = "IEEE Robotics and Automation Society advances innovation in robotics."
        vec = self.embedding_mgr.encode_query(text)
        self.assertEqual(vec.shape, (1, EMBEDDING_DIM))
        # Normalized vector norm should be approximately 1.0
        import numpy as np
        norm = np.linalg.norm(vec[0])
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_03_relevant_retrieval_ieee_ras(self):
        """Verify retrieval for 'What is IEEE RAS?' returns relevant chunks with valid metadata."""
        query = "What is IEEE RAS?"
        res = self.retriever.retrieve(query, top_k=3)
        self.assertTrue(res["is_relevant"], "Query should be marked relevant.")
        self.assertGreater(res["max_score"], 0.40, "Similarity score should be high.")
        self.assertGreater(len(res["chunks"]), 0, "Should return at least 1 chunk.")
        
        # Check metadata fields
        chunk = res["chunks"][0]
        self.assertIn("title", chunk)
        self.assertIn("url", chunk)
        self.assertIn("text", chunk)
        self.assertTrue(chunk["url"].startswith("http"), f"Invalid URL: {chunk['url']}")

    def test_04_relevant_retrieval_committees(self):
        """Verify retrieval for technical committees."""
        query = "What are the main technical committees in IEEE RAS?"
        res = self.retriever.retrieve(query, top_k=3)
        self.assertTrue(res["is_relevant"])
        # At least one chunk should mention technical committees or specific committees
        combined_text = " ".join([c["text"] for c in res["chunks"]])
        self.assertTrue(
            "Technical Committees" in combined_text or "Robotics" in combined_text,
            "Technical committees context not retrieved."
        )

    def test_05_relevant_retrieval_conferences(self):
        """Verify retrieval for conferences (ICRA, IROS, CASE)."""
        query = "What conferences are associated with IEEE RAS?"
        res = self.retriever.retrieve(query, top_k=3)
        self.assertTrue(res["is_relevant"])
        combined_text = " ".join([c["text"] for c in res["chunks"]])
        self.assertTrue(
            "ICRA" in combined_text or "Conferences" in combined_text,
            "Conference context not retrieved."
        )

    def test_06_out_of_domain_rejection(self):
        """Verify unrelated queries (e.g., 'What is the capital of France?') are flagged as out of scope."""
        query = "What is the capital of France?"
        # Execute RAG answer
        result = self.rag_pipeline.answer_question(query)
        self.assertFalse(result["is_relevant"], "Unrelated query should not be marked relevant.")
        self.assertIn("does not contain verified records", result["answer"])

    def test_07_empty_question_handling(self):
        """Verify empty or whitespace-only questions are handled gracefully."""
        result = self.rag_pipeline.answer_question("   ")
        self.assertIn("Please ask a question", result["answer"])


if __name__ == "__main__":
    unittest.main()

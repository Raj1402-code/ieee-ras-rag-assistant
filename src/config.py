"""
Configuration and settings for the IEEE RAS AI Knowledge Assistant.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
VECTORSTORE_DIR = BASE_DIR / "vectorstore"

# Vector database paths
FAISS_INDEX_PATH = VECTORSTORE_DIR / "index.faiss"
METADATA_PATH = VECTORSTORE_DIR / "metadata.json"

# RAG & Embedding settings
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
TOP_K_RETRIEVAL = 4
SIMILARITY_THRESHOLD = 0.30  # Min cosine similarity threshold to consider relevant

# LLM settings
DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Chunking parameters
CHUNK_SIZE = 750
CHUNK_OVERLAP = 120

# System prompt enforcing strict anti-hallucination and grounded citations
RAG_SYSTEM_INSTRUCTION = """You are an IEEE Robotics and Automation Society (IEEE RAS) information assistant.
Answer the user's question using ONLY the retrieved IEEE RAS context provided below.

Rules:
1. Do not invent, assume, or extrapolate any IEEE RAS facts, committees, conferences, people, dates, awards, or links.
2. If the retrieved context does not contain enough information to answer the question accurately, state clearly and politely:
   "The available IEEE RAS public records do not provide enough information to answer this question."
3. If the user asks a question completely unrelated to IEEE RAS (e.g., general geography, pop culture, or unrelated math), explicitly state that you can only provide information regarding the IEEE Robotics and Automation Society based on available records.
4. Structure your response professionally using clean markdown (bullet points, clear paragraphs, and bold section headings).
5. At the very end of your answer, provide a "### Sources" section with numbered markdown links using the exact URLs from the retrieved context. Format:
   - [1] [Page Title](URL)
   - [2] [Page Title](URL)
6. Do not fabricate URLs or reference sources that were not provided in the context.
"""

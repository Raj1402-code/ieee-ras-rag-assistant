"""
RAG pipeline integrating semantic retrieval with Google Gemini LLM generation.
"""

import os
import logging
from typing import Dict, Any, List, Optional
from google import genai
from google.genai import types

from src.config import (
    DEFAULT_GEMINI_MODEL,
    GEMINI_API_KEY,
    RAG_SYSTEM_INSTRUCTION
)
from src.retriever import get_retriever
from src.utils import deduplicate_sources

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Orchestrates retrieval and grounded response generation."""

    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "") or GEMINI_API_KEY
        self.model_name = model_name or os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)
        self.retriever = get_retriever()
        self._client: Optional[genai.Client] = None
        self._init_client()

    def _init_client(self):
        """Initialize Google GenAI client if API key is present."""
        if self.api_key and self.api_key.strip():
            try:
                self._client = genai.Client(api_key=self.api_key.strip())
            except Exception as e:
                logger.error("Failed to initialize GenAI client: %s", e)
                self._client = None
        else:
            self._client = None

    def update_api_key(self, new_key: str):
        """Update API key dynamically from UI."""
        self.api_key = new_key
        self._init_client()

    def has_valid_key(self) -> bool:
        """Check if an API key is configured."""
        return bool(self._client is not None and self.api_key and len(self.api_key.strip()) > 10)

    def _build_context_prompt(self, query: str, chunks: List[Dict[str, Any]]) -> str:
        """Construct prompt combining query and retrieved IEEE RAS chunks."""
        context_blocks = []
        for i, chunk in enumerate(chunks, 1):
            title = chunk.get("title", "IEEE RAS Resource")
            url = chunk.get("url", "https://www.ieee-ras.org/")
            text = chunk.get("text", "").strip()
            score = chunk.get("similarity_score", 0.0)
            context_blocks.append(
                f"--- [Document {i}] ---\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Relevance Score: {score}\n"
                f"Content:\n{text}\n"
            )
        
        context_str = "\n".join(context_blocks)
        
        prompt = f"""RETRIEVED IEEE RAS CONTEXT:
========================
{context_str}
========================

USER QUESTION:
{query}

INSTRUCTIONS FOR ANSWER:
1. Provide an accurate, comprehensive, and well-structured answer using ONLY facts from the retrieved IEEE RAS context above.
2. If the context does not explicitly support an answer, clearly explain: "The available IEEE RAS public records do not provide sufficient information to answer this question."
3. At the end of your response, list the sources cited with exact clickable markdown links in a "### Sources" section.
"""
        return prompt

    def answer_question(self, query: str, chat_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline:
        Retrieve -> Verify Relevance -> Format Context -> Call Gemini -> Return Answer & Citations.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            return {
                "answer": "Please ask a question about IEEE RAS.",
                "sources": [],
                "retrieved_chunks": [],
                "is_relevant": False,
                "similarity_score": 0.0
            }

        # Step 1: Vector similarity retrieval
        retrieval_res = self.retriever.retrieve(cleaned_query)
        chunks = retrieval_res.get("chunks", [])
        is_relevant = retrieval_res.get("is_relevant", False)
        max_score = retrieval_res.get("max_score", 0.0)

        # Step 2: Out-of-Domain Guardrail
        # If retriever found no matches or similarity is lower than threshold
        if not is_relevant or not chunks:
            out_of_scope_msg = (
                f"The available IEEE RAS public knowledge base does not contain verified records "
                f"regarding your query: *\"{cleaned_query}\"*.\n\n"
                f"This assistant is strictly trained to answer questions regarding the "
                f"**IEEE Robotics and Automation Society (IEEE RAS)**, its technical committees, "
                f"conferences (e.g., ICRA, IROS), publications (e.g., T-RO, RA-L), student branches, "
                f"membership, and educational activities.\n\n"
                f"Please try asking an IEEE RAS-related question, or choose an example from the sidebar."
            )
            return {
                "answer": out_of_scope_msg,
                "sources": [],
                "retrieved_chunks": chunks,
                "is_relevant": False,
                "similarity_score": max_score
            }

        unique_sources = deduplicate_sources(chunks)

        # Step 3: Check API Key readiness
        if not self.has_valid_key():
            # Return context summary without LLM call if key is missing
            warning_msg = (
                "⚠️ **Gemini API Key Missing**: A Google Gemini API key (`GEMINI_API_KEY`) is required to generate natural language answers.\n\n"
                "**Retrieved IEEE RAS Context Summary**:\n\n"
            )
            for s in unique_sources[:3]:
                warning_msg += f"- **[{s['title']}]({s['url']})**: {s['excerpt']}\n\n"
            warning_msg += "\n*Please provide your `GEMINI_API_KEY` in your environment or via the sidebar to enable full RAG synthesis.*"
            return {
                "answer": warning_msg,
                "sources": unique_sources,
                "retrieved_chunks": chunks,
                "is_relevant": True,
                "similarity_score": max_score
            }

        # Step 4: Call Gemini LLM with context
        prompt = self._build_context_prompt(cleaned_query, chunks)

        # List of candidate models to try in order
        candidate_models = [self.model_name]
        for fallback in ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-1.5-flash"]:
            if fallback not in candidate_models:
                candidate_models.append(fallback)

        config = types.GenerateContentConfig(
            system_instruction=RAG_SYSTEM_INSTRUCTION,
            temperature=0.15,  # Low temperature for strict factual grounding
            max_output_tokens=1200
        )

        last_error = None
        for model in candidate_models:
            try:
                logger.info("Attempting generation with model: %s", model)
                response = self._client.models.generate_content(
                    model=model,
                    contents=prompt,
                    config=config
                )

                generated_answer = response.text if response and response.text else "Unable to generate a response from the model."
                self.model_name = model  # Lock in the working model

                return {
                    "answer": generated_answer,
                    "sources": unique_sources,
                    "retrieved_chunks": chunks,
                    "is_relevant": True,
                    "similarity_score": max_score,
                    "model_used": model
                }

            except Exception as e:
                err_msg = str(e)
                logger.warning("Generation with model %s failed: %s", model, err_msg)
                last_error = e
                # If it's a 404 NOT_FOUND, try the next model in the candidate list
                if "404" in err_msg or "NOT_FOUND" in err_msg or "not found" in err_msg.lower():
                    continue
                else:
                    # For quota or auth issues, break immediately
                    break

        logger.error("All Gemini model attempts failed. Last error: %s", last_error)
        return {
            "answer": f"An error occurred while generating the response from Gemini: `{str(last_error)}`.\n\nPlease verify your API key quota or network connectivity.",
            "sources": unique_sources,
            "retrieved_chunks": chunks,
            "is_relevant": True,
            "similarity_score": max_score,
            "error": str(last_error)
        }


_rag_pipeline_instance = None


def get_rag_pipeline(api_key: Optional[str] = None) -> RAGPipeline:
    """Singleton getter for RAG pipeline."""
    global _rag_pipeline_instance
    if _rag_pipeline_instance is None:
        _rag_pipeline_instance = RAGPipeline(api_key=api_key)
    elif api_key and api_key != _rag_pipeline_instance.api_key:
        _rag_pipeline_instance.update_api_key(api_key)
    return _rag_pipeline_instance

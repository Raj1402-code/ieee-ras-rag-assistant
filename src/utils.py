"""
Utility functions for text processing, chunking, and source formatting.
"""

import re
from typing import List, Dict, Any


def clean_text(text: str) -> str:
    """Normalize whitespace and remove unwanted characters."""
    if not text:
        return ""
    # Replace carriage returns and excessive tabs
    text = text.replace("\r", " ").replace("\t", " ")
    # Replace multiple spaces with a single space
    text = re.sub(r" +", " ", text)
    # Replace 3 or more consecutive newlines with 2 newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def recursive_chunk_text(
    text: str,
    chunk_size: int = 750,
    chunk_overlap: int = 120
) -> List[str]:
    """
    Split text into overlapping chunks respecting paragraph and sentence boundaries.
    """
    cleaned = clean_text(text)
    if not cleaned:
        return []
    
    if len(cleaned) <= chunk_size:
        return [cleaned]
    
    # Split by double newline (paragraphs) first
    paragraphs = cleaned.split("\n\n")
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # If adding this paragraph keeps chunk within limit
        if len(current_chunk) + len(para) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{para}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            
            # If a single paragraph is larger than chunk_size, split by sentences
            if len(para) > chunk_size:
                sentences = re.split(r"(?<=[.!?])\s+", para)
                current_chunk = ""
                for sent in sentences:
                    sent = sent.strip()
                    if not sent:
                        continue
                    if len(current_chunk) + len(sent) + 1 <= chunk_size:
                        current_chunk = f"{current_chunk} {sent}".strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        
                        # Handle very long sentences
                        if len(sent) > chunk_size:
                            for i in range(0, len(sent), chunk_size - chunk_overlap):
                                chunks.append(sent[i : i + chunk_size].strip())
                            current_chunk = ""
                        else:
                            current_chunk = sent
            else:
                # Retain overlap from previous chunk if possible
                overlap_text = current_chunk[-chunk_overlap:] if len(current_chunk) >= chunk_overlap else ""
                current_chunk = f"{overlap_text}\n\n{para}".strip() if overlap_text else para

    if current_chunk and (not chunks or current_chunk != chunks[-1]):
        chunks.append(current_chunk)
    
    # Filter any empty chunks
    return [c.strip() for c in chunks if c.strip()]


def deduplicate_sources(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Return unique sources based on URL while preserving highest similarity score."""
    seen_urls = {}
    for chunk in chunks:
        url = chunk.get("url", "").strip()
        if not url:
            continue
        score = chunk.get("similarity_score", 0.0)
        if url not in seen_urls or score > seen_urls[url].get("similarity_score", 0.0):
            seen_urls[url] = {
                "title": chunk.get("title", "IEEE RAS Resource"),
                "url": url,
                "category": chunk.get("category", "General"),
                "similarity_score": score,
                "excerpt": chunk.get("text", "")[:180] + "..."
            }
    return list(seen_urls.values())

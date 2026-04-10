# ingestion/pyramid.py
# Enriches each text chunk with a summary, category, and keywords.
# This forms the "knowledge pyramid" — multiple representations of the same text.

import re
from typing import List, Dict


# Keywords that indicate a legal document
LEGAL_KEYWORDS = {
    "agreement", "contract", "party", "parties", "clause",
    "liability", "jurisdiction", "indemnify", "statute", "defendant",
    "plaintiff", "whereas", "hereinafter", "obligation", "breach",
}


def build_pyramid(chunks: List[str]) -> List[Dict]:
    """
    Build a knowledge pyramid entry for each chunk.

    Args:
        chunks: List of raw text chunks.

    Returns:
        List of dicts, each with: raw_text, summary, category, keywords.
    """
    pyramid = []
    for chunk in chunks:
        entry = {
            "raw_text": chunk,
            "summary": _extract_summary(chunk),
            "category": _classify_category(chunk),
            "keywords": _extract_keywords(chunk),
        }
        pyramid.append(entry)
    return pyramid


def _extract_summary(text: str) -> str:
    """Return the first two sentences as a summary."""
    # Split on sentence-ending punctuation followed by whitespace or end-of-string
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    # Take up to the first 2 sentences
    summary_sentences = sentences[:2]
    return " ".join(summary_sentences).strip()


def _classify_category(text: str) -> str:
    """
    Rule-based category classification:
    - 'math'  : if the text contains digits/numbers
    - 'legal' : if the text contains known legal keywords
    - 'general': otherwise
    """
    text_lower = text.lower()

    # Check for legal vocabulary
    words_in_text = set(re.findall(r'\b\w+\b', text_lower))
    if words_in_text & LEGAL_KEYWORDS:
        return "legal"

    # Check for presence of numbers (digits)
    if re.search(r'\d', text):
        return "math"

    return "general"


def _extract_keywords(text: str, top_n: int = 5) -> List[str]:
    """
    Extract the top N most frequent meaningful words.
    Filters out common stop words for better signal.
    """
    STOP_WORDS = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at",
        "to", "for", "of", "with", "is", "it", "this", "that",
        "was", "are", "be", "by", "as", "from", "have", "has",
        "not", "we", "i", "you", "he", "she", "they", "their",
        "our", "its", "if", "so", "do", "can", "will", "more",
    }

    # Lowercase and tokenize
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())

    # Count word frequencies, excluding stop words
    freq: Dict[str, int] = {}
    for word in words:
        if word not in STOP_WORDS:
            freq[word] = freq.get(word, 0) + 1

    # Sort by frequency descending and return top N
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, _ in sorted_words[:top_n]]

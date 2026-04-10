# ingestion/sliding_window.py
# Splits raw text into overlapping chunks using a sliding window approach.
# This ensures no context is lost at chunk boundaries.

from typing import List


def sliding_window_chunks(
    text: str,
    window_size: int = 2000,
    overlap: int = 500,
) -> List[str]:
    """
    Split text into overlapping character-based chunks.

    Args:
        text:        The raw input text to chunk.
        window_size: Number of characters per chunk (default 2000).
        overlap:     Number of characters to repeat between consecutive chunks (default 500).

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    if window_size <= overlap:
        raise ValueError("window_size must be greater than overlap.")

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + window_size
        chunk = text[start:end]
        chunks.append(chunk)

        # Move the window forward by (window_size - overlap) characters
        # so the next chunk shares 'overlap' characters with the current one
        start += window_size - overlap

    return chunks

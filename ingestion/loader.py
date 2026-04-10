# ingestion/loader.py
# Loads a PDF or plain text file and returns raw text content.

import os


def load_document(file_path: str) -> str:
    """
    Load a PDF or .txt file and return its raw text.

    Args:
        file_path: Path to the document.

    Returns:
        Raw text string extracted from the document.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".txt":
        return _load_text(file_path)
    elif ext == ".pdf":
        return _load_pdf(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .txt or .pdf")


def _load_text(file_path: str) -> str:
    """Read a plain text file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def _load_pdf(file_path: str) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    Falls back to a clear error if the library is missing.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF support. Install it with: pip install pymupdf"
        )

    text_parts = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())

    return "\n".join(text_parts)

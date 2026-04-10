# bonus/router.py
# Simple reasoning-aware query router.
# Decides which domain pipeline (math / legal / general) should handle a query.
# Design is intentionally keyword-based and stateless for clarity and speed.
# In production this would be replaced by a fine-tuned intent classifier.

import re
from typing import Literal

# Type alias for the three supported route types
RouteType = Literal["math", "legal", "general"]

# Legal vocabulary used for classification
LEGAL_TERMS = {
    "contract", "agreement", "liability", "indemnify", "clause",
    "jurisdiction", "statute", "plaintiff", "defendant", "breach",
    "parties", "obligation", "hereinafter", "lawsuit", "legal",
    "attorney", "court", "appeal", "damages", "remedy",
}


def route_query(query: str) -> RouteType:
    """
    Classify a query into one of three routes.

    Rules (applied in order of priority):
      1. 'legal'   — if any known legal keyword appears in the query.
      2. 'math'    — if the query contains digits or math operators (+, -, *, /).
      3. 'general' — fallback for everything else.

    Args:
        query: The raw user query string.

    Returns:
        One of: 'math', 'legal', 'general'
    """
    query_lower = query.lower()
    words_in_query = set(re.findall(r'\b\w+\b', query_lower))

    # Priority 1: legal
    if words_in_query & LEGAL_TERMS:
        return "legal"

    # Priority 2: math — digits or arithmetic operators
    if re.search(r'\d', query) or re.search(r'[+\-*/=]', query):
        return "math"

    # Fallback
    return "general"


def describe_route(route: RouteType) -> str:
    """Return a human-readable description of what the route does."""
    descriptions = {
        "math":    "Route to the mathematical reasoning pipeline (GSM8K model).",
        "legal":   "Route to the legal document analysis pipeline.",
        "general": "Route to the general-purpose document retrieval pipeline.",
    }
    return descriptions[route]


# ── Standalone demo ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    test_queries = [
        "What is 25 + 37?",
        "Explain the indemnity clause in the agreement.",
        "Who wrote the novel 1984?",
        "If a train travels 60 miles per hour for 3 hours, how far does it go?",
        "What are the obligations of both parties in the contract?",
        "Tell me about climate change.",
    ]

    print("=== Query Router Demo ===\n")
    for q in test_queries:
        route = route_query(q)
        print(f"Query   : {q}")
        print(f"Route   : {route}")
        print(f"Action  : {describe_route(route)}")
        print()

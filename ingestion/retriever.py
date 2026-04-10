# ingestion/retriever.py
# Retrieves the most relevant pyramid chunk for a given query.
# Compares the query against raw_text, summary, and keywords using TF-IDF cosine similarity.

from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def retrieve(query: str, pyramid: List[Dict]) -> Dict:
    """
    Find and return the pyramid entry most similar to the query.

    Strategy:
    - For each pyramid entry, build a combined representation:
        raw_text + summary + space-joined keywords
    - Vectorize all representations + the query using TF-IDF
    - Return the entry with the highest cosine similarity to the query

    Args:
        query:   The user's search query.
        pyramid: List of pyramid dicts (output of pyramid.build_pyramid).

    Returns:
        The best matching pyramid entry dict.
    """
    if not pyramid:
        raise ValueError("Pyramid is empty — ingest a document first.")

    # Build one combined text per pyramid entry for richer matching
    corpus = []
    for entry in pyramid:
        keywords_text = " ".join(entry.get("keywords", []))
        combined = f"{entry['raw_text']} {entry['summary']} {keywords_text}"
        corpus.append(combined)

    # Add the query as the last document so we can compare in one matrix
    all_texts = corpus + [query]

    # Fit TF-IDF vectorizer on the full corpus (including query)
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    # The query vector is the last row
    query_vector = tfidf_matrix[-1]
    # Corpus vectors are all rows except the last
    corpus_vectors = tfidf_matrix[:-1]

    # Compute cosine similarity between query and each corpus entry
    similarities = cosine_similarity(query_vector, corpus_vectors).flatten()

    # Return the entry with the highest similarity score
    best_index = similarities.argmax()
    best_entry = pyramid[best_index]

    # Attach similarity score for transparency (optional but useful)
    best_entry["_score"] = round(float(similarities[best_index]), 4)
    return best_entry

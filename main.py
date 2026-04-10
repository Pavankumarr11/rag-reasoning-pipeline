# main.py
# Entry point for the Document Ingestion + Knowledge Pyramid pipeline.
# Flow:
#   1. Load a document (PDF or .txt)
#   2. Split into overlapping chunks via sliding window
#   3. Enrich each chunk into a pyramid entry (summary, category, keywords)
#   4. Accept a query from the user
#   5. Retrieve the best-matching chunk
#   6. Print the result

import sys

from ingestion.loader        import load_document
from ingestion.sliding_window import sliding_window_chunks
from ingestion.pyramid        import build_pyramid
from ingestion.retriever      import retrieve
from bonus.router             import route_query, describe_route


def run_ingestion_pipeline(file_path: str, query: str):
    """
    Run the full document ingestion and retrieval pipeline.

    Args:
        file_path : Path to the document (.txt or .pdf).
        query     : The search query to answer from the document.
    """
    print("=" * 60)
    print("DOCUMENT INGESTION & RETRIEVAL PIPELINE")
    print("=" * 60)

    # ── Step 1: Load document ──────────────────────────────────────
    print(f"\n[1] Loading document: {file_path}")
    text = load_document(file_path)
    print(f"    Loaded {len(text):,} characters.")

    # ── Step 2: Sliding window chunking ───────────────────────────
    print("\n[2] Applying sliding window chunking...")
    chunks = sliding_window_chunks(text, window_size=2000, overlap=500)
    print(f"    Created {len(chunks)} chunks.")

    # ── Step 3: Build knowledge pyramid ───────────────────────────
    print("\n[3] Building knowledge pyramid...")
    pyramid = build_pyramid(chunks)
    print(f"    Pyramid entries: {len(pyramid)}")

    # Show a sample entry summary
    print(f"\n    Sample pyramid entry (chunk 0):")
    entry0 = pyramid[0]
    print(f"      Category  : {entry0['category']}")
    print(f"      Keywords  : {entry0['keywords']}")
    print(f"      Summary   : {entry0['summary'][:120]}...")

    # ── Step 4: Route the query ────────────────────────────────────
    print(f"\n[4] Routing query: '{query}'")
    route = route_query(query)
    print(f"    Route   : {route}")
    print(f"    Action  : {describe_route(route)}")

    # ── Step 5: Retrieve best matching chunk ───────────────────────
    print("\n[5] Retrieving best matching chunk...")
    best = retrieve(query, pyramid)

    # ── Step 6: Print results ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("RESULT")
    print("=" * 60)
    print(f"Similarity Score : {best.get('_score', 'N/A')}")
    print(f"Category         : {best['category']}")
    print(f"Keywords         : {', '.join(best['keywords'])}")
    print(f"\nSummary:\n  {best['summary']}")
    print(f"\nFull Chunk (first 500 chars):\n  {best['raw_text'][:500]}...")
    print("=" * 60)


def create_sample_document(path: str = "sample_doc.txt"):
    """Create a small sample .txt file for demo purposes."""
    content = """
    Artificial Intelligence and Machine Learning

    Machine learning is a subset of artificial intelligence. It enables computers to learn from data
    without being explicitly programmed. In 2023, large language models became widely available.
    These models are trained on billions of tokens and can generate human-like text.

    Mathematics plays a central role in ML. Gradient descent minimises the loss function by computing
    derivatives. A simple linear model has 2 parameters: weight and bias. Training on 10,000 examples
    over 50 epochs typically reduces loss from 2.3 to below 0.5 for classification tasks.

    Legal and Ethical Considerations

    The European AI Act is a landmark regulation that establishes obligations for AI providers.
    High-risk AI systems must comply with strict liability clauses. The agreement between developers
    and deployers sets out each party's obligations. Jurisdictions differ in their approach to AI liability.
    Breach of these obligations may result in substantial damages.

    Natural Language Processing

    Tokenisation splits text into smaller units called tokens. Transformers use attention mechanisms
    to relate tokens across long contexts. BERT uses masked language modelling while GPT uses causal
    language modelling. Both architectures have achieved state-of-the-art results on numerous benchmarks.
    Fine-tuning a pre-trained model requires far fewer labelled examples than training from scratch.
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"Sample document created: {path}")
    return path


if __name__ == "__main__":
    # If a file path is provided as a CLI argument, use it; otherwise create a demo file.
    if len(sys.argv) >= 2:
        doc_path = sys.argv[1]
        query    = sys.argv[2] if len(sys.argv) >= 3 else "What is machine learning?"
    else:
        print("No file path provided — creating a sample document for demo.\n")
        doc_path = create_sample_document()
        query    = "How does gradient descent work in machine learning?"

    run_ingestion_pipeline(doc_path, query)

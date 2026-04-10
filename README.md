# AI Reasoning System — Document Ingestion + LoRA Training + Query Router

A clean, modular Python project implementing:

1. **Document Ingestion System** — Sliding Window + Knowledge Pyramid
2. **GSM8K Reasoning Model Training** — LoRA fine-tuning on distilgpt2
3. **Bonus: Reasoning-Aware Router** — rule-based query classifier

---

## Project Structure

```
PROJECT_V2/
├── ingestion/
│   ├── loader.py          # Load PDF or .txt files
│   ├── sliding_window.py  # Character-based sliding window chunker
│   ├── pyramid.py         # Build summary / category / keywords per chunk
│   └── retriever.py       # TF-IDF cosine similarity retrieval
├── training/
│   ├── dataset.py         # Load & format GSM8K dataset
│   ├── model.py           # distilgpt2 + LoRA adapter setup
│   ├── train.py           # Training loop
│   └── evaluate.py        # Exact-match accuracy evaluation
├── bonus/
│   └── router.py          # Route queries: math / legal / general
├── main.py                # Full ingestion pipeline demo
├── requirements.txt
└── README.md
```

---

## Setup

**1. Create a virtual environment (recommended)**
```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Running the Ingestion Pipeline

**Demo with auto-generated sample document:**
```bash
python main.py
```

**With your own .txt file:**
```bash
python main.py path/to/your_document.txt "Your query here"
```

**With a PDF:**
```bash
python main.py path/to/document.pdf "What does this document say about contracts?"
```

The pipeline will:
1. Load and chunk the document (2000-char windows, 500-char overlap)
2. Build a knowledge pyramid (summary + category + keywords per chunk)
3. Route the query (math / legal / general)
4. Retrieve the most relevant chunk using TF-IDF cosine similarity
5. Print the result

---

## Running GSM8K Training

```bash
python -m training.train
```

This will:
- Download the GSM8K dataset (first run only)
- Load distilgpt2 with LoRA adapters
- Train for 1 epoch on 3000 samples
- Save the model to `gsm8k_lora_model/`

---

## Running Evaluation

After training:
```bash
python -m training.evaluate
```

Outputs exact-match accuracy comparing predicted final answers to ground-truth answers.

---

## Running the Query Router (Bonus)

```bash
python -m bonus.router
```

Or import directly:
```python
from bonus.router import route_query
print(route_query("What is 25 + 37?"))        # → math
print(route_query("Explain the liability clause"))  # → legal
print(route_query("Who wrote Hamlet?"))        # → general
```

---

## Design Notes

| Component | Approach | Why |
|---|---|---|
| Chunking | Sliding window (char-based) | Simple, no tokenizer dependency |
| Pyramid | Rule-based classification | Fast, interpretable, no model needed |
| Retrieval | TF-IDF cosine similarity | No vector DB required, works offline |
| Training | LoRA on distilgpt2 | Efficient; degrades gracefully without PEFT |
| Router | Keyword matching | Zero latency, fully explainable |

---

## ⚠️ Limitations

* Uses TF-IDF instead of semantic embeddings (may miss contextual similarity)
* Router is rule-based (not ML-based)
* Limited training samples (3000 GSM8K subset)
* No UI or API layer yet

---

## 🚀 Future Improvements

* Replace TF-IDF with vector embeddings (FAISS / ChromaDB)
* Upgrade router using LLM-based classification
* Add FastAPI endpoint for production usage
* Integrate RAG pipeline with LLM responses
* Deploy on cloud (AWS / Hugging Face Spaces)

---


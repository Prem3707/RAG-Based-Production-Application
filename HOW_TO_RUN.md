# ▶️ How to Run — Production RAG

## Prerequisites
- Python 3.11+
- OpenAI API key

## Setup

```bash
# 1. Navigate to project
cd 01-production-rag

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

## Step 1 — Add Your Documents
```bash
# Drop PDFs or .txt files into data/docs/
cp your_document.pdf data/docs/
```

## Step 2 — Ingest Documents
```bash
python -m src.ingest
# Output: "Ingestion complete." — creates chroma_db and bm25_index.pkl
```

## Step 3 — Run the API
```bash
python -m src.api.app
# Server at http://localhost:8000
```

## Step 4 — Ask a Question
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the main findings?"}'
```

## Step 5 — Run Evaluation
```bash
# Add your ground-truth questions to ci/eval_questions.json first
python -m src.evaluation.eval_pipeline
# Exit code 0 = all thresholds passed; 1 = failure
```

## CI (GitHub Actions)
Push to GitHub and the eval pipeline runs automatically on every PR.
Add `OPENAI_API_KEY` as a GitHub Actions Secret in repo Settings → Secrets.

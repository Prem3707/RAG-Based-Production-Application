# 🔍 Project 1: Production RAG Application

## What Is It?
A domain-specific "Ask My Docs" system — you feed it your PDFs or text files and it answers natural-language questions about them with **cited, verifiable answers**. Every claim in the answer links back to the exact document chunk it came from.

---

## Industrial Applications
| Industry | Use Case |
|---|---|
| Legal | Query contracts, case law, regulatory docs |
| Healthcare | Ask clinical guidelines, drug databases |
| Finance | Interrogate earnings reports, compliance docs |
| Engineering | Search technical manuals, datasheets |
| Enterprise | Internal knowledge base search |

---

## Architecture

```
User Question
     │
     ▼
┌─────────────────────────────────────┐
│           Hybrid Retriever          │
│  ┌───────────────┐ ┌─────────────┐  │
│  │ Vector Search │ │ BM25 Search │  │
│  │  (ChromaDB)   │ │  (keyword)  │  │
│  └───────┬───────┘ └──────┬──────┘  │
│          └────────┬───────┘         │
│               RRF Fusion            │
│                   │                 │
│          Cross-Encoder Reranker     │
└───────────────────┬─────────────────┘
                    │ Top-4 chunks
                    ▼
            GPT-4o-mini (LLM)
            Citation enforced
                    │
                    ▼
           Answer + [chunk_id] citations
```

---

## Key Technical Concepts

### BM25 (Best Match 25)
Keyword-based ranking algorithm. Excellent for **exact term matches** ("GDPR Article 17"). Fails on synonyms.

### Vector Search (ChromaDB)
Semantic similarity via embeddings. Great for **meaning-based** queries ("data deletion rights"). Fails on rare proper nouns.

### RRF Fusion
Combines ranked lists from both methods — a chunk ranks high if it appears near the top in *either* list. This beats both methods alone in most benchmarks.

### Cross-Encoder Reranking
A second ML model that scores each (query, chunk) pair jointly — much more accurate than embedding similarity but too slow to run on thousands of chunks. Run it only on the top ~10 fused candidates.

### Citation Enforcement
The prompt **requires** the LLM to use `[chunk_id]` tags. Post-generation validation checks that valid IDs appear — if not, the request is rejected with a `CitationError`. This prevents hallucinated "citations."

---

## File Structure
```
01-production-rag/
├── src/
│   ├── ingest.py                    # Document loading → chunking → indexing
│   ├── rag_chain.py                 # Main RAG pipeline + citation validator
│   ├── retrieval/
│   │   └── hybrid_retriever.py      # BM25 + vector + RRF + reranking
│   ├── evaluation/
│   │   └── eval_pipeline.py         # RAGAS metrics + CI gate
│   └── api/
│       └── app.py                   # FastAPI REST endpoint
├── data/docs/                       # ← Put your PDFs/TXTs here
├── ci/
│   └── eval_questions.json          # Ground-truth QA pairs
├── .github/workflows/eval.yml       # CI pipeline
└── requirements.txt
```

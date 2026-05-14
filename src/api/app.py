"""
FastAPI REST API for the Production RAG System.
Exposes /query endpoint and /health check.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import time

from src.rag_chain import RAGChain, CitationError


app = FastAPI(
    title="Production RAG API",
    description="Domain-specific Ask My Docs system with hybrid retrieval",
    version="1.0.0",
)

# Initialize once at startup
rag_chain = None


@app.on_event("startup")
async def startup_event():
    global rag_chain
    rag_chain = RAGChain()
    print("[API] RAG chain loaded and ready.")


class QueryRequest(BaseModel):
    question: str
    enforce_citations: bool = True


class SourceItem(BaseModel):
    chunk_id: str
    source: str
    page: Optional[int]
    snippet: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    latency_ms: float


@app.get("/health")
def health():
    return {"status": "ok", "model": "gpt-4o-mini"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest):
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG chain not initialized")

    start = time.time()
    try:
        result = rag_chain.run(
            request.question,
            enforce_citations=request.enforce_citations,
        )
    except CitationError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {e}")

    latency_ms = (time.time() - start) * 1000

    return QueryResponse(
        question=result["question"],
        answer=result["answer"],
        sources=[SourceItem(**s) for s in result["sources"]],
        latency_ms=round(latency_ms, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.app:app", host="0.0.0.0", port=8000, reload=True)

"""
Hybrid Retriever: fuses BM25 (keyword) + ChromaDB (semantic) results
using Reciprocal Rank Fusion (RRF), then reranks with a cross-encoder.
"""

import pickle
from pathlib import Path
from typing import List, Tuple, Dict, Any

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from sentence_transformers import CrossEncoder


VECTOR_STORE_DIR = Path("data/chroma_db")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Reciprocal Rank Fusion constant
RRF_K = 60


class HybridRetriever:
    def __init__(self, top_k: int = 10, rerank_top_n: int = 4):
        self.top_k = top_k
        self.rerank_top_n = rerank_top_n

        # Load vector store
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.vector_store = Chroma(
            persist_directory=str(VECTOR_STORE_DIR),
            embedding_function=embeddings,
        )

        # Load BM25 index
        with open(BM25_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        self.bm25 = data["bm25"]
        self.bm25_chunks = data["chunks"]

        # Load cross-encoder reranker
        self.reranker = CrossEncoder(RERANKER_MODEL)
        print(f"[Retriever] Ready — top_k={top_k}, rerank_top_n={rerank_top_n}")

    def _vector_search(self, query: str) -> List[Tuple[Any, float]]:
        """Semantic vector search, returns (doc, score) pairs."""
        results = self.vector_store.similarity_search_with_score(query, k=self.top_k)
        return results  # [(Document, score), ...]

    def _bm25_search(self, query: str) -> List[Tuple[Any, float]]:
        """Keyword BM25 search, returns (doc, score) pairs."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(self.bm25_chunks[i], score) for i, score in ranked[:self.top_k]]

    def _rrf_fusion(
        self,
        vector_results: List[Tuple[Any, float]],
        bm25_results: List[Tuple[Any, float]],
    ) -> List[Any]:
        """Reciprocal Rank Fusion to merge two ranked lists."""
        chunk_scores: Dict[str, float] = {}
        chunk_map: Dict[str, Any] = {}

        for rank, (doc, _) in enumerate(vector_results):
            uid = doc.metadata.get("chunk_id", doc.page_content[:20])
            chunk_scores[uid] = chunk_scores.get(uid, 0) + 1 / (RRF_K + rank + 1)
            chunk_map[uid] = doc

        for rank, (doc, _) in enumerate(bm25_results):
            uid = doc.metadata.get("chunk_id", doc.page_content[:20])
            chunk_scores[uid] = chunk_scores.get(uid, 0) + 1 / (RRF_K + rank + 1)
            chunk_map[uid] = doc

        sorted_uids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)
        return [chunk_map[uid] for uid in sorted_uids[:self.top_k]]

    def _rerank(self, query: str, candidates: List[Any]) -> List[Any]:
        """Use cross-encoder to rerank fused candidates."""
        pairs = [(query, doc.page_content) for doc in candidates]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        return [doc for _, doc in ranked[:self.rerank_top_n]]

    def retrieve(self, query: str) -> List[Any]:
        """Full hybrid retrieval pipeline."""
        vector_results = self._vector_search(query)
        bm25_results = self._bm25_search(query)
        fused = self._rrf_fusion(vector_results, bm25_results)
        reranked = self._rerank(query, fused)
        return reranked

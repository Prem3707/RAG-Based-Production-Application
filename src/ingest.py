"""
Document ingestion pipeline for Production RAG.
Loads documents, chunks them, embeds via vector store, and indexes for BM25.
"""

import os
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
)
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from rank_bm25 import BM25Okapi
import pickle


DATA_DIR = Path("data/docs")
VECTOR_STORE_DIR = Path("data/chroma_db")
BM25_INDEX_PATH = Path("data/bm25_index.pkl")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64


def load_documents(directory: Path) -> List[Any]:
    """Load all PDFs and text files from a directory."""
    docs = []
    for file_path in directory.rglob("*.pdf"):
        loader = PyPDFLoader(str(file_path))
        docs.extend(loader.load())
    for file_path in directory.rglob("*.txt"):
        loader = TextLoader(str(file_path))
        docs.extend(loader.load())
    print(f"[Ingest] Loaded {len(docs)} raw document chunks")
    return docs


def chunk_documents(documents: List[Any]) -> List[Any]:
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " "],
    )
    chunks = splitter.split_documents(documents)
    # Attach a stable chunk ID for citation tracing
    for i, chunk in enumerate(chunks):
        source = chunk.metadata.get("source", "unknown")
        page = chunk.metadata.get("page", 0)
        uid = hashlib.md5(f"{source}-{page}-{i}".encode()).hexdigest()[:8]
        chunk.metadata["chunk_id"] = uid
    print(f"[Ingest] Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    return chunks


def build_vector_store(chunks: List[Any]) -> Chroma:
    """Embed chunks and store in ChromaDB."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(VECTOR_STORE_DIR),
    )
    vector_store.persist()
    print(f"[Ingest] Vector store saved to {VECTOR_STORE_DIR}")
    return vector_store


def build_bm25_index(chunks: List[Any]) -> BM25Okapi:
    """Build a BM25 keyword index over the same chunks."""
    tokenized = [chunk.page_content.lower().split() for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump({"bm25": bm25, "chunks": chunks}, f)
    print(f"[Ingest] BM25 index saved to {BM25_INDEX_PATH}")
    return bm25


def run_ingestion():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    documents = load_documents(DATA_DIR)
    if not documents:
        print("[Ingest] No documents found — add PDFs or .txt files to data/docs/")
        return
    chunks = chunk_documents(documents)
    build_vector_store(chunks)
    build_bm25_index(chunks)
    print("[Ingest] ✅ Ingestion complete.")


if __name__ == "__main__":
    run_ingestion()

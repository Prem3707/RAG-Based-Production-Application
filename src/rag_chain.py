"""
RAG Chain: wraps the hybrid retriever with an LLM that is forced to
cite every claim using [chunk_id] notation. Raises CitationError if
the model returns an answer without citations.
"""

import re
from typing import List, Dict, Any

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document

from src.retrieval.hybrid_retriever import HybridRetriever


SYSTEM_PROMPT = """You are a precise question-answering assistant.
You MUST support every factual claim with an inline citation in the format [chunk_id].
If you cannot find evidence in the provided context, say "I don't have enough information."
Never fabricate facts. Never answer without citations.

Context chunks (use chunk_id from metadata):
{context}
"""

USER_PROMPT = "{question}"


class CitationError(Exception):
    pass


def format_context(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{chunk_id}] (source: {source})\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def validate_citations(answer: str, docs: List[Document]) -> bool:
    """Check that at least one valid chunk_id appears in the answer."""
    valid_ids = {doc.metadata.get("chunk_id") for doc in docs}
    found = set(re.findall(r"\[([a-f0-9]{8})\]", answer))
    return bool(found & valid_ids)


class RAGChain:
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.retriever = HybridRetriever()
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            ("human", USER_PROMPT),
        ])

    def run(self, question: str, enforce_citations: bool = True) -> Dict[str, Any]:
        # Step 1: Retrieve
        docs = self.retriever.retrieve(question)

        # Step 2: Build context
        context = format_context(docs)

        # Step 3: Generate
        chain = self.prompt | self.llm
        response = chain.invoke({"context": context, "question": question})
        answer = response.content

        # Step 4: Validate citations
        if enforce_citations and not validate_citations(answer, docs):
            raise CitationError(
                f"Model returned answer without valid citations.\nAnswer: {answer}"
            )

        return {
            "question": question,
            "answer": answer,
            "sources": [
                {
                    "chunk_id": d.metadata.get("chunk_id"),
                    "source": d.metadata.get("source"),
                    "page": d.metadata.get("page"),
                    "snippet": d.page_content[:200],
                }
                for d in docs
            ],
        }


if __name__ == "__main__":
    rag = RAGChain()
    result = rag.run("What are the key findings of the report?")
    print(result["answer"])
    print("\nSources:")
    for s in result["sources"]:
        print(f"  [{s['chunk_id']}] {s['source']} p.{s['page']}")

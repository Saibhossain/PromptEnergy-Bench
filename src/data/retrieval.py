"""Retrieval Module for Experiment 3 (RAG Extension).

Provides abstract retriever interface and a lightweight, fully reproducible
pure-Python BM25 implementation over GSM8K TRAIN corpus.
"""

import math
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Set, Optional
from src.data.gsm8k import GSM8KRecord, load_gsm8k
from src.data.context_builder import estimate_tokens


@dataclass
class RetrievalResult:
    retriever_name: str
    top_k: int
    retrieval_latency_ms: float
    retrieved_document_ids: List[str]
    retrieved_context_tokens: int
    context_text: str


class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        """Retrieves top_k documents for a given query."""
        pass


class BM25Retriever(BaseRetriever):
    """Okapi BM25 Retriever implemented in pure Python for zero-dependency portability."""

    def __init__(self, corpus: Optional[List[GSM8KRecord]] = None, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        if corpus is None:
            self.corpus = load_gsm8k(split="train")
        else:
            self.corpus = corpus

        self.doc_ids = [doc.id for doc in self.corpus]
        self.doc_texts = [f"Question: {doc.question}\nSolution: {doc.solution}\nAnswer: {doc.answer}" for doc in self.corpus]
        self.doc_tokens = [self._tokenize(text) for text in self.doc_texts]
        self.doc_lens = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_len = sum(self.doc_lens) / max(1, len(self.doc_lens))
        self.num_docs = len(self.corpus)

        # Compute document frequencies (DF)
        self.df: Dict[str, int] = {}
        for tokens in self.doc_tokens:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                self.df[token] = self.df.get(token, 0) + 1

        # Compute IDF
        self.idf: Dict[str, float] = {}
        for token, freq in self.df.items():
            self.idf[token] = math.log((self.num_docs - freq + 0.5) / (freq + 0.5) + 1.0)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        start_time = time.perf_counter()
        query_tokens = self._tokenize(query)

        scores = [0.0] * self.num_docs
        for q_token in query_tokens:
            if q_token not in self.idf:
                continue
            idf_val = self.idf[q_token]
            for doc_idx, tokens in enumerate(self.doc_tokens):
                tf = tokens.count(q_token)
                if tf > 0:
                    doc_len = self.doc_lens[doc_idx]
                    num = tf * (self.k1 + 1)
                    denom = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                    scores[doc_idx] += idf_val * (num / denom)

        ranked_indices = sorted(range(self.num_docs), key=lambda i: scores[i], reverse=True)[:top_k]
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        retrieved_ids = [self.doc_ids[idx] for idx in ranked_indices]
        retrieved_snippets = []
        for idx in ranked_indices:
            doc = self.corpus[idx]
            retrieved_snippets.append(
                f"[Retrieved Example {doc.id}]:\nProblem: {doc.question}\nSolution: {doc.solution}\nAnswer: {doc.answer}\n"
            )

        context_text = "Retrieved Mathematical Knowledge:\n" + "\n".join(retrieved_snippets)
        actual_tokens = estimate_tokens(context_text)

        return RetrievalResult(
            retriever_name="bm25",
            top_k=top_k,
            retrieval_latency_ms=round(latency_ms, 3),
            retrieved_document_ids=retrieved_ids,
            retrieved_context_tokens=actual_tokens,
            context_text=context_text
        )


class OptionalEmbeddingRetriever(BaseRetriever):
    """Stub interface for dense vector embedding retrieval."""
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name

    def retrieve(self, query: str, top_k: int = 3) -> RetrievalResult:
        raise NotImplementedError("OptionalEmbeddingRetriever requires torch/sentence-transformers. Use BM25Retriever for zero-dependency baseline.")

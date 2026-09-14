"""Unit tests for BM25 Retrieval and RAG extension."""

import unittest
from src.data.gsm8k import GSM8KRecord
from src.data.retrieval import BM25Retriever, RetrievalResult
from src.infrastructure.checkpoint import compute_condition_key


class TestBM25Retriever(unittest.TestCase):

    def setUp(self):
        self.mock_corpus = [
            GSM8KRecord(
                id="gsm8k_train_0001",
                question="John buys 5 apples for $2 each. How much does he pay?",
                answer="10",
                solution="5 * 2 = 10",
                raw_answer="5 * 2 = 10\n#### 10"
            ),
            GSM8KRecord(
                id="gsm8k_train_0002",
                question="Mary has 12 oranges. She bakes cakes using 4 oranges each. How many cakes can she bake?",
                answer="3",
                solution="12 / 4 = 3",
                raw_answer="12 / 4 = 3\n#### 3"
            ),
            GSM8KRecord(
                id="gsm8k_train_0003",
                question="An airplane travels 500 miles per hour for 3 hours. How far does it travel?",
                answer="1500",
                solution="500 * 3 = 1500",
                raw_answer="500 * 3 = 1500\n#### 1500"
            )
        ]
        self.retriever = BM25Retriever(corpus=self.mock_corpus)

    def test_retrieve_relevant_document(self):
        query = "How much for 3 apples at $2 each?"
        res = self.retriever.retrieve(query=query, top_k=1)
        self.assertIsInstance(res, RetrievalResult)
        self.assertEqual(res.top_k, 1)
        self.assertIn("gsm8k_train_0001", res.retrieved_doc_ids)
        self.assertGreater(res.context_tokens, 0)
        self.assertGreaterEqual(res.retrieval_latency_ms, 0.0)

    def test_top_k_multi_retrieval(self):
        query = "Mary buys apples and oranges for cakes"
        res = self.retriever.retrieve(query=query, top_k=2)
        self.assertEqual(len(res.retrieved_doc_ids), 2)
        self.assertEqual(res.top_k, 2)

    def test_exclude_id_prevents_contamination(self):
        query = "John buys 5 apples for $2 each."
        # Exclude doc 1
        res = self.retriever.retrieve(query=query, top_k=1, exclude_id="gsm8k_train_0001")
        self.assertNotIn("gsm8k_train_0001", res.retrieved_doc_ids)

    def test_top_k_zero_returns_empty(self):
        res = self.retriever.retrieve(query="What is 2+2?", top_k=0)
        self.assertEqual(res.top_k, 0)
        self.assertEqual(res.retrieved_doc_ids, [])
        self.assertEqual(res.context_tokens, 0)
        self.assertEqual(res.context_text, "")

    def test_condition_key_for_rag(self):
        key1 = compute_condition_key(
            experiment_name="rag_gsm8k",
            sample_id="gsm8k_test_0001",
            model="qwen3.5:0.8b",
            strategy="rag_top_3",
            repetition=1,
            context_type="bm25_retrieval",
            retriever="bm25",
            top_k=3
        )
        key2 = compute_condition_key(
            experiment_name="rag_gsm8k",
            sample_id="gsm8k_test_0001",
            model="qwen3.5:0.8b",
            strategy="rag_top_1",
            repetition=1,
            context_type="bm25_retrieval",
            retriever="bm25",
            top_k=1
        )
        self.assertEqual(len(key1), 24)
        self.assertNotEqual(key1, key2)
        # Deterministic check
        self.assertEqual(
            key1,
            compute_condition_key(
                experiment_name="rag_gsm8k",
                sample_id="gsm8k_test_0001",
                model="qwen3.5:0.8b",
                strategy="rag_top_3",
                repetition=1,
                context_type="bm25_retrieval",
                retriever="bm25",
                top_k=3
            )
        )


if __name__ == "__main__":
    unittest.main()

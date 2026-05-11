#!/usr/bin/env python3
"""
retriever.py — Hybrid retrieval with RRF fusion and cross-encoder reranking.

Pipeline per query:
  1. Embed query → dense vector (Ollama mxbai-embed-large)
  2. Encode query → sparse BM25 vector (vocabulary rebuilt from Qdrant payloads)
  3. Prefetch dense + sparse candidates, merge with Qdrant-native RRF fusion
  4. Rerank top-k with CrossEncoder, return the single best result
"""

import math
import re
from collections import Counter
from dataclasses import dataclass

import ollama
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

COLLECTION = "kb_hybrid"
DENSE_MODEL = "mxbai-embed-large"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_CANDIDATES = 20  # candidates fetched per branch before fusion
TOP_K = 5            # results passed to cross-encoder


@dataclass
class RetrievalResult:
    text: str
    source: str
    rrf_score: float
    ce_score: float


# ---------------------------------------------------------------------------
# Shared tokenizer (must match ingestion.py exactly)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


# ---------------------------------------------------------------------------
# Corpus vocabulary — rebuilt at init from stored Qdrant payloads
# ---------------------------------------------------------------------------

def _build_vocab(client: QdrantClient, collection: str) -> tuple[dict, dict]:
    """
    Scroll all payload texts from `collection` and rebuild the same
    vocabulary + IDF that ingestion.py computed from the corpus.

    Returns
    -------
    vocab : {word: int_index}   — sorted, matches ingestion indices
    idf   : {word: float}       — Robertson-Jones IDF, same formula as fastbm25
    """
    texts: list[str] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        texts.extend(p.payload["text"] for p in points)
        if offset is None:
            break

    tokenized = [tokenize(t) for t in texts]
    all_terms = {term for doc in tokenized for term in doc}
    vocab: dict[str, int] = {word: idx for idx, word in enumerate(sorted(all_terms))}

    N = len(tokenized)
    df = Counter(term for doc in tokenized for term in set(doc))
    idf: dict[str, float] = {
        term: math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1.0)
        for term in vocab
    }
    return vocab, idf


# ---------------------------------------------------------------------------
# HybridRetriever
# ---------------------------------------------------------------------------

class HybridRetriever:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        dense_model: str = DENSE_MODEL,
        cross_encoder_model: str = CROSS_ENCODER_MODEL,
        collection: str = COLLECTION,
        top_k: int = TOP_K,
    ) -> None:
        self.client = QdrantClient(url=qdrant_url)
        self.dense_model = dense_model
        self.collection = collection
        self.top_k = top_k

        print("Loading cross-encoder…")
        self.cross_encoder = CrossEncoder(cross_encoder_model)

        print("Rebuilding corpus vocabulary from Qdrant payloads…")
        self._vocab, self._idf = _build_vocab(self.client, collection)
        print(f"Vocabulary ready ({len(self._vocab)} terms).")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query_text: str) -> RetrievalResult | None:
        """
        Run hybrid search + RRF fusion + cross-encoder reranking.
        Returns the single best-matching RetrievalResult, or None.
        """
        dense_vec = self._embed_dense(query_text)
        sp_idx, sp_val = self._embed_sparse(query_text)

        prefetch: list[Prefetch] = [
            Prefetch(query=dense_vec, using="dense", limit=RRF_CANDIDATES),
        ]
        if sp_idx:
            prefetch.append(
                Prefetch(
                    query=SparseVector(indices=sp_idx, values=sp_val),
                    using="sparse",
                    limit=RRF_CANDIDATES,
                )
            )

        candidates = self.client.query_points(
            collection_name=self.collection,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.top_k,
            with_payload=True,
        ).points

        if not candidates:
            return None

        return self._rerank(query_text, candidates)

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    def _embed_dense(self, text: str) -> list[float]:
        return ollama.embeddings(model=self.dense_model, prompt=text)["embedding"]

    def _embed_sparse(self, text: str) -> tuple[list[int], list[float]]:
        """
        Query-side BM25 sparse vector.

        Document vectors store IDF * TF_norm(t, d), so the query weight for
        term t is 1.0 per unique occurrence — the dot product then yields the
        standard BM25 score: Σ_t IDF(t) * TF_norm(t, d).
        """
        seen: set[str] = set()
        indices: list[int] = []
        values: list[float] = []
        for token in tokenize(text):
            if token in self._vocab and token not in seen:
                indices.append(self._vocab[token])
                values.append(1.0)
                seen.add(token)
        return indices, values

    # ------------------------------------------------------------------
    # Cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank(self, query_text: str, candidates: list) -> RetrievalResult:
        pairs = [[query_text, c.payload["text"]] for c in candidates]
        ce_scores = self.cross_encoder.predict(pairs)

        best_idx = int(ce_scores.argmax())
        best = candidates[best_idx]

        return RetrievalResult(
            text=best.payload["text"],
            source=best.payload.get("source", ""),
            rrf_score=best.score,
            ce_score=float(ce_scores[best_idx]),
        )

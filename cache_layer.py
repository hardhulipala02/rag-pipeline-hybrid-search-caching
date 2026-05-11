#!/usr/bin/env python3
"""
cache_layer.py — Semantic cache backed by Qdrant.

SemanticCache stores (query embedding → response) pairs. On a cache hit
(cosine similarity > SIMILARITY_THRESHOLD) it returns the saved response
instead of calling the LLM again.
"""

import uuid

import ollama
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

COLLECTION = "llm_cache"
DENSE_MODEL = "mxbai-embed-large"
DENSE_DIM = 1024
SIMILARITY_THRESHOLD = 0.95


class SemanticCache:
    def __init__(
        self,
        qdrant_url: str = "http://localhost:6333",
        model: str = DENSE_MODEL,
        threshold: float = SIMILARITY_THRESHOLD,
    ) -> None:
        self.client = QdrantClient(url=qdrant_url)
        self.model = model
        self.threshold = threshold
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, query_text: str) -> str | None:
        """Return a cached response for query_text, or None on a miss."""
        vector = self._embed(query_text)
        results = self.client.query_points(
            collection_name=COLLECTION,
            query=vector,
            score_threshold=self.threshold,
            limit=1,
        ).points
        if results:
            return results[0].payload["response"]
        return None

    def set(self, query_text: str, response_text: str) -> None:
        """Embed query_text and store the (query, response) pair."""
        vector = self._embed(query_text)
        self.client.upsert(
            collection_name=COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "query": query_text,
                        "response": response_text,
                    },
                )
            ],
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> list[float]:
        return ollama.embeddings(model=self.model, prompt=text)["embedding"]

    def _ensure_collection(self) -> None:
        existing = {c.name for c in self.client.get_collections().collections}
        if COLLECTION not in existing:
            self.client.create_collection(
                collection_name=COLLECTION,
                vectors_config=VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
            )

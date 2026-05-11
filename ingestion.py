#!/usr/bin/env python3
"""
ingestion.py — Load documents from ./data, embed with Ollama + BM25, upsert to Qdrant.

Dense vector  : Ollama mxbai-embed-large  (named "dense",  1024-dim, cosine)
Sparse vector : BM25 term weights         (named "sparse", via fastbm25)
Collection    : kb_hybrid
"""

import re
import uuid
from pathlib import Path

import ollama
from fastbm25 import fastbm25
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseIndexParams,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

DATA_DIR = Path("./data")
COLLECTION = "kb_hybrid"
DENSE_MODEL = "mxbai-embed-large"
DENSE_DIM = 1024


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_documents(data_dir: Path) -> list[dict]:
    docs = []
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            docs.append({"id": str(uuid.uuid4()), "text": text, "source": path.name})
    return docs


# ---------------------------------------------------------------------------
# Dense embeddings
# ---------------------------------------------------------------------------

def embed_dense(texts: list[str]) -> list[list[float]]:
    vectors = []
    for i, text in enumerate(texts, 1):
        resp = ollama.embeddings(model=DENSE_MODEL, prompt=text)
        vectors.append(resp["embedding"])
        print(f"  [{i}/{len(texts)}] dense embedding for '{text[:60]}…'" if len(text) > 60
              else f"  [{i}/{len(texts)}] dense embedding done")
    return vectors


# ---------------------------------------------------------------------------
# Sparse vectors (BM25)
# ---------------------------------------------------------------------------

def tokenize(text: str) -> list[str]:
    return re.sub(r"[^a-z0-9\s]", "", text.lower()).split()


def build_sparse_vectors(
    texts: list[str],
) -> list[tuple[list[int], list[float]]]:
    tokenized = [tokenize(t) for t in texts]

    # fastbm25 computes BM25 weights and stores them in document_score:
    #   {word: {doc_idx: bm25_score}}
    model = fastbm25(tokenized)

    # Build a stable word → integer index mapping from the scored vocabulary
    vocab: dict[str, int] = {
        word: idx for idx, word in enumerate(sorted(model.document_score))
    }

    sparse: list[tuple[list[int], list[float]]] = []
    for doc_idx in range(len(texts)):
        indices: list[int] = []
        values: list[float] = []
        for word, doc_scores in model.document_score.items():
            if doc_idx in doc_scores:
                indices.append(vocab[word])
                values.append(float(doc_scores[doc_idx]))
        sparse.append((indices, values))

    return sparse


# ---------------------------------------------------------------------------
# Qdrant
# ---------------------------------------------------------------------------

def ensure_collection(client: QdrantClient) -> None:
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        print(f"Collection '{COLLECTION}' already exists, reusing.")
        return

    client.create_collection(
        collection_name=COLLECTION,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )
    print(f"Created collection '{COLLECTION}'.")


def upsert(
    client: QdrantClient,
    docs: list[dict],
    dense_vecs: list[list[float]],
    sparse_vecs: list[tuple[list[int], list[float]]],
) -> None:
    points = [
        PointStruct(
            id=doc["id"],
            vector={
                "dense": dense,
                "sparse": SparseVector(indices=sp_idx, values=sp_val),
            },
            payload={"text": doc["text"], "source": doc["source"]},
        )
        for doc, dense, (sp_idx, sp_val) in zip(docs, dense_vecs, sparse_vecs)
    ]
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"Upserted {len(points)} point(s) into '{COLLECTION}'.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"Data directory '{DATA_DIR}' not found.")

    print(f"Loading documents from '{DATA_DIR}'…")
    docs = load_documents(DATA_DIR)
    if not docs:
        print("No documents found. Add text files to ./data and re-run.")
        return
    print(f"Loaded {len(docs)} document(s).\n")

    print("Generating dense embeddings via Ollama…")
    dense_vecs = embed_dense([d["text"] for d in docs])

    print("\nComputing BM25 sparse vectors…")
    sparse_vecs = build_sparse_vectors([d["text"] for d in docs])
    print(f"  Sparse vectors ready ({len(sparse_vecs)} docs).\n")

    client = QdrantClient(url="http://localhost:6333")
    ensure_collection(client)
    upsert(client, docs, dense_vecs, sparse_vecs)


if __name__ == "__main__":
    main()

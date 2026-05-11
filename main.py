#!/usr/bin/env python3
"""
main.py — Interactive RAG loop with semantic caching.

Flow per question:
  1. SemanticCache.get    →  return immediately on hit
  2. HybridRetriever      →  hybrid search + RRF + cross-encoder rerank
  3. Ollama llama3.1      →  generate grounded answer (streamed to terminal)
  4. SemanticCache.set    →  store answer for future identical/similar queries
"""

import sys

import ollama

from cache_layer import SemanticCache
from retriever import HybridRetriever

LLM_MODEL = "llama3.1:latest"
DIVIDER = "─" * 60

SYSTEM_PROMPT = (
    "You are a financial research assistant specialising in SEC filings. "
    "Answer the user's question using ONLY the context provided. "
    "If the context does not contain enough information, say so clearly and briefly."
)


# ---------------------------------------------------------------------------
# LLM call — streams tokens to stdout, returns the full answer string
# ---------------------------------------------------------------------------

def _generate(question: str, context: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer based only on the context above:"
            ),
        },
    ]
    parts: list[str] = []
    for chunk in ollama.chat(model=LLM_MODEL, messages=messages, stream=True):
        token = chunk["message"]["content"]
        print(token, end="", flush=True)
        parts.append(token)
    print()
    return "".join(parts)


# ---------------------------------------------------------------------------
# Single-question handler
# ---------------------------------------------------------------------------

def _handle(question: str, cache: SemanticCache, retriever: HybridRetriever) -> None:
    print()

    # 1. Cache check
    cached = cache.get(question)
    if cached:
        print(DIVIDER)
        print(f"  ✓ Cache hit\n")
        print(cached)
        print(DIVIDER)
        print()
        return

    # 2. Hybrid retrieval
    print("  Retrieving context…")
    result = retriever.retrieve(question)

    if result is None:
        print("  No relevant documents found. Try rephrasing your question.\n")
        return

    print(DIVIDER)
    print(f"  Source : {result.source}")
    print(f"  RRF    : {result.rrf_score:.4f}   CE : {result.ce_score:.4f}")
    print()

    # 3. LLM generation (streamed)
    answer = _generate(question, result.text)

    # 4. Store in cache
    cache.set(question, answer)

    print(DIVIDER)
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("\nInitialising cache…")
    cache = SemanticCache()

    print("Initialising retriever…")
    retriever = HybridRetriever()

    print()
    print("  SEC Filings RAG  ·  Hybrid Search + Semantic Cache")
    print(DIVIDER)
    print("  Type your question, or 'exit' / 'quit' to stop.")
    print(DIVIDER)
    print()

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            sys.exit(0)

        if not question:
            continue
        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            sys.exit(0)

        _handle(question, cache, retriever)


if __name__ == "__main__":
    main()

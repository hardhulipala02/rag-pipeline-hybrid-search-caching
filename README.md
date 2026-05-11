# 📊 Hybrid RAG Pipeline with Semantic Caching

An intelligent financial data assistant that combines **Hybrid Search** with **Semantic Caching** to provide fast, accurate insights into corporate SEC filings while minimizing redundant AI processing.

## 🚀 Overview
This system transforms raw, structured financial data into a searchable "knowledge base" using a multi-stage RAG (Retrieval-Augmented Generation) architecture. By utilizing a high-speed **Semantic Cache**, the agent remembers previous questions and answers, offering instant responses for common queries while reserving heavy LLM computation for new, complex insights.

## 🛠️ Technical Architecture
The system is built as a **layered intelligence stack**:

1.  **Ingestion:** Converts SEC CSV data into natural language narratives and generates dual embeddings: **Dense** (for conceptual similarity) and **Sparse** (for exact keyword matching).
2.  **Memory Shield (The Cache):** Employs a Qdrant-backed semantic cache with a **0.95 similarity threshold** to intercept repeat queries and deliver answers in milliseconds.
3.  **Hybrid Retrieval:** Simultaneously executes dual-vector searches across the knowledge base to ensure coverage of both thematic concepts and specific technical identifiers like CIK numbers.
4.  **Consensus Ranking (RRF):** Utilizes **Reciprocal Rank Fusion** to mathematically merge Dense and Sparse search results, prioritizing documents that appear consistently across both methods.
5.  **Quality Control (Reranking):** Uses a **Cross-Encoder** (`ms-marco-MiniLM-L-6-v2`) to perform deep-dive analysis on the top 5 results, ensuring only the single most relevant context reaches the LLM.
6.  **Inference:** Sends the verified context and user query to **Llama 3.1** via **Ollama** to synthesize a grounded, professional response.
7.  **Auto-Caching:** Automatically saves the generated response back to the semantic cache to optimize the system for future identical or similar queries.

## 🧰 Tech Stack
* **Language:** Python 3.10+
* **AI Orchestration:** Ollama (Llama 3.1 & mxbai-embed-large)
* **Vector Database:** Qdrant (Running in Docker with volume persistence)
* **Reranking:** Sentence-Transformers Cross-Encoder
* **Environment:** MacOS / Linux

## 🛡️ Key Features
* **Semantic Awareness:** Goes beyond exact word matching to understand the "vibe" and intent of financial queries.
* **Extreme Performance:** Achieves ~10-50ms response times for cached queries, significantly reducing local hardware strain.
* **Persistent Knowledge:** Uses Docker volume mapping to ensure the vector database and cache survive container restarts.
* **Production-Grade Precision:** The combination of Hybrid Search and Cross-Encoder reranking minimizes "hallucinations" by providing the LLM with highly accurate context.

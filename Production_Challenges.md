# Production Challenges & Interview Guide: Hybrid Recommender System

This document outlines the technical challenges, architectural trade-offs, and typical interview questions related to the **Hybrid Recommender System**. Use this to prepare for deep technical discussions during your interview.

---

## 1. Production-Level Challenges

### A. Scalability & Vector Search
*   **Current State**: Uses a FAISS "Flat" index. This is an exact search that computes the distance between the query and *every* vector in the dataset.
*   **Production Issue**: With 389 assessments, it's fast. With 10 million assessments, the latency would be unacceptable.
*   **Production Solution**: Move to **Approximate Nearest Neighbor (ANN)** algorithms like **HNSW** (Hierarchical Navigable Small World) or **IVF** (Inverted File Index). These trade a tiny bit of accuracy for massive speed gains.

### B. LLM Latency & Reliability
*   **Current State**: Uses a local LLM (Qwen) for query planning.
*   **Production Issue**: As seen on Hugging Face CPU, this can take 60+ seconds. In a real app, users expect <200ms.
*   **Production Solution**:
    *   Use **GPU acceleration** (CUDA/MPS).
    *   Use a **slimmer model** or a **fine-tuned distilled model**.
    *   Implement **aggressive caching** (Semantic Cache): If two queries are semantically similar, reuse the previous LLM plan.
    *   **Fallback logic**: If the LLM takes >500ms, fall back to the deterministic BM25 query.

### C. Reranker Overhead
*   **Current State**: Reranks the top-200 candidates using a Cross-Encoder.
*   **Production Issue**: Cross-Encoders are computationally expensive because they process the query and document *together* in the transformer. 200 documents is a lot for a synchronous API call.
*   **Production Solution**:
    *   **Tiered Reranking**: Use a fast Bi-Encoder (Cosine Similarity) for the top-1000, and only use the Cross-Encoder for the final top-20.
    *   **Quantization**: Use INT8 or ONNX versions of the reranker to speed up inference.

### D. Data Freshness (The "Cold Start" Problem)
*   **Current State**: Static catalog crawled and indexed once.
*   **Production Issue**: If SHL adds new assessments today, they won't appear until you re-run the whole pipeline.
*   **Production Solution**: Implement an **Incremental Indexing** pipeline. When new data arrives:
    1.  Crawl the individual page.
    2.  Compute its embedding.
    3.  Add it to the FAISS index without rebuilding the whole thing.

---

## 2. System Design Questions

### Q1: "Why use a Hybrid (BM25 + Vector) approach instead of just Vector search?"
**Answer**:
*   **Vector Search** is great for *semantic* meaning (e.g., "collaboration" matching "teamwork") but can struggle with exact keywords, acronyms, or specific test IDs.
*   **BM25 (Lexical Search)** is excellent at keyword matching.
*   **Hybrid** gives you the best of both worlds. We use **RRF (Reciprocal Rank Fusion)** to combine them, ensuring that if a result is strong in either (or both) methods, it surfaces to the top.

### Q2: "How would you evaluate if this system is actually 'good'?"
**Answer**:
*   **Offline Evaluation**:
    *   **Hit Rate / Recall@K**: If we have labeled data of what users eventually chose, does it show up in our top 10?
    *   **NDCG (Normalized Discounted Cumulative Gain)**: Does the *best* result appear at rank 1?
*   **Online Evaluation (A/B Testing)**:
    *   **CTR (Click-Through Rate)**: Do users click the recommendations?
    *   **Conversion Rate**: Do users actually start the assessments we recommend?

### Q3: "The LLM generates a 'Query Plan'. Is that really necessary? Why not just embed the user query directly?"
**Answer**:
*   Raw user queries are often messy: *"I need a test for a junior java dev 40 mins"*.
*   The **Query Planner** (LLM) extracts structured metadata like `duration=40` and `job_level=junior`. This allows us to apply **Hard Constraints** (filtering) alongside the **Soft Semantic Search**.
*   It also performs **Query Expansion**: translating "Java Dev" into related concepts like "coding logic," "syntax," or "problem solving," which improves retrieval coverage.

### Q4: "What happens if the LLM 'hallucinates' a skill that isn't in the job description?"
**Answer**:
*   This is a risk. We mitigate this by:
    1.  **System Prompting**: Restricting the LLM to only extract concepts present in the text.
    2.  **Schema Validation**: Using Pydantic to ensure the LLM output follows a strict structure.
    3.  **Safety Net**: The Reranker act as a final "sanity check." If the LLM expands into a hallucinated area, the Reranker (which sees the original query and the actual document) will assign a low score to those irrelevant results.

---

## 3. High-Level Architecture (Brief)

If asked to draw the system:
1.  **Crawl Layer**: Playwright extracts raw HTML.
2.  **Indexing Layer**: BGE-small embeddings + FAISS (Dense) & BM25 (Sparse).
3.  **LLM Planner**: Transforms raw query into `retrieval_query` + `constraints`.
4.  **Retrieval Layer**: Parallel search in BM25 & FAISS $\rightarrow$ RRF Fusion.
5.  **Rerank Layer**: MiniLM Cross-Encoder picks the final top 10.
6.  **Constraint Layer**: Final filtering (e.g., "Max 40 minutes").

---

## 4. Key Metrics to mention
*   **Catalog Size**: ~390 assessments.
*   **Embedding Model**: `BAAI/bge-small-en-v1.5` (optimized for speed/memory).
*   **Reranker**: `cross-encoder/ms-marco-MiniLM-L-6-v2` (industry standard for RAG).
*   **Fusion Algorithm**: RRF ($k=60$).

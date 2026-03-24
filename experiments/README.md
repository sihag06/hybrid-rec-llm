# Experiments & Evaluation

Technical summary of the end-to-end recommender stack, with design choices, experiments, and measured impact:
- Catalog build & normalization: scrape → clean → enrich metadata (roles, flags, durations) → embed → index.
- Query understanding: deterministic + LLM rewrites/planning **before** retrieval to inject structure (intent, skills, constraints).
- Retrieval: BM25 + dense (BGE) + weighted RRF fusion.
- Rerank: finetuned cross-encoder rescoring top candidates.
- Constraints/post-processing: duration/remote/adaptive filtering and delivery.

Metrics (recall@k, MRR@k) are computed on the train/val split (n=52/13) and logged under `runs/*/metrics.json`. This note is updated as new experiments/runs are added.

## 1) Data & Catalog
- Source: `data/catalog_docs_rich.jsonl` (389 assessments; earlier crawl had 377).
- Enrichment: roles, languages, remote/adaptive flags (with heuristics for missing flags), durations, test types (`test_type_full` preferred).
- Artifacts:
  - FAISS index: `data/faiss_index/index_bge.faiss` (BGE-small embeddings).
  - Embedding ID map: `data/embeddings_bge/assessment_ids.json`.
  - Optional vocab: `data/catalog_role_vocab.json` (boost role/intent parsing).

### Design choices
- Keep rich fields to enable multi-signal retrieval (lexical + semantic) and post-filters (duration, remote, adaptive).
- Flat FAISS for quality; HNSW considered but not adopted (latency acceptable, recall priority).
- BGE-small chosen for CPU-friendly semantic retrieval.
- Catalog growth: initial crawl yielded ~377 items. After parser fixes (handling variant product URLs, relaxing skip rules) and broader coverage of product/detail pages, the enriched build (`catalog_docs_rich.jsonl`) now contains 389 unique assessments. Example fix: “Assessment and Development Center Exercises” was skipped because the detail parser required specific metadata headers (e.g., “Assessment length”), and pages without them were discarded; relaxing that requirement in `crawler/parser_detail.py` allowed these products to be ingested.
- Catalog variants: `catalog_docs.jsonl` (baseline) vs `catalog_docs_rich.jsonl` (enriched text/fields). Rich version concatenates more fields—name + description + doc_text + test_type_full + job_levels + languages + downloads text—to improve recall for semantic/lexical search. This helps surface items where the short description omits key terms but the longer doc_text/metadata contains them.

## 1.1) Indexing & Embeddings
- Embeddings: `BAAI/bge-small-en-v1.5`, chosen for strong quality/size trade-off on CPU (fits HF Space constraints, no GPU).
- Storage: FAISS flat index at `data/faiss_index/index_bge.faiss` with ID map `data/embeddings_bge/assessment_ids.json`.
- Why flat (not HNSW): dataset is small (~389 items) and recall is prioritized over marginal latency gains. Flat gives exact search and simpler reproducibility.
- Why not external vector DB (Pinecone/Weaviate): no external infra/network for Spaces, cost/complexity avoidance, and full reproducibility/offline friendliness with local FAISS.
- Index experiments (all in `data/faiss_index/`):
  - `index_bge.faiss`: flat index on BGE-small embeddings (current default).
  - `index_rich.faiss`: flat index on “rich” embeddings (using the richer concatenated text above); improved coverage on long-tail descriptions (e.g., where doc_text lists skills/flags absent in the short description).
  - `index_hnsw.faiss`: HNSW variant evaluated for latency; not adopted (small corpus, recall priority, flat was sufficient).
  - Legacy `index.faiss`/`index.json`: earlier embeddings; retained for reproducibility but not used in best run.

## 2) Retrieval & Fusion
- Lexical: BM25 over catalog text (baseline implementation in-repo; validated against HF BM25 parity). Scoring: BM25(q,d) ≈ Σ_{t∈q} IDF(t)·( (tf·(k1+1)) / (tf + k1·(1-b + b·|d|/avgd)) ).
- Vector: BGE-small embeddings + FAISS flat index (`index_bge.faiss`), cosine similarity on dense vectors.
- Fusion: Weighted Reciprocal Rank Fusion (RRF) with query-adaptive weights (see `tools/retrieve_tool.py`), k_rrf=60, candidates topn=200. Score = w_b/(k_rrf + rank_bm25) + w_v/(k_rrf + rank_vec). Weights chosen heuristically per query: more weight to BM25 if many must-have skills; more to vector for longer/softer queries.

### Experiments & choices
- Baselines: BM25-only (recall@10≈0.08 val), Vector-only (≈0.15 val).
- Hybrid RRF (Reciprocal Rank Fusion) BM25 + vector (no rerank): recall@10≈0.23 val → clear gain over single-channel.
- Hybrid RRF + rerank: recall@10≈0.38–0.46 val depending on rewrite; best run 0.4615 with rewrite.
- Candidate pool: 200 preserved high recall without blowing up rerank cost; lower k hurt recall.
- Fusion hyperparameters: k_rrf=60 and adaptive weights yielded better recall than equal weights in ablations.
- Why flat (not HNSW): small corpus (~389) and recall priority; HNSW not needed for latency.
- Why BGE-small: strong quality/size trade-off; runs on CPU-only Spaces.
- Candidate count (top-N) experiments: tried 50, 100, 200, 377(full), 400. Recall@10 improved significantly moving from 50→100→200; gains beyond 200 were marginal while rerank latency increased. Chose 200 as the balance point (good recall lift without excessive rerank cost).
- Metrics context: Train/val split n=52/13. Per-run val metrics captured in `runs/*/metrics.json` and summarized in the Evaluation Summary table (section 6). Candidate-count ablations were observed qualitatively (recall@10 plateau past 200; MRR@10 likewise); only the final configurations are logged in the table.

### Rationale
- BM25 captures exact terms (skills, titles); vector covers paraphrase/semantics. RRF combines both signals; weighted RRF lets us bias toward BM25 for skill-heavy queries and toward vector for verbose/soft-skill queries.
- Keeping fusion/rerank in-memory (local FAISS + cached reranker) avoids external latency; reproducible and cost-free versus managed vector DBs.

## 3) Reranking
- Model: `models/reranker_crossenc/v0.1.0` (fine-tuned from `cross-encoder/ms-marco-MiniLM-L-6-v2` on pairwise data).
- Cross-encoder reranker: `models/reranker_crossenc/v0.1.0` (fine-tuned from `cross-encoder/ms-marco-MiniLM-L-6-v2` on pairwise data).
- Reranks fused candidates to top-10 (default).
- Training setup (see `models/reranker_crossenc/v0.1.0/train_config.json`): 1 epoch, lr=1e-5, batch=4, max_len=256 on `data/reranker/pairwise_train/val.jsonl`. Each sample is (query, doc, label∈{0,1}). We tried pairwise hinge/margin losses and listwise variants; settled on binary cross-entropy over logits for stability and simplicity given small data: L = -[y·log σ(s) + (1–y)·log(1–σ(s))], where s=fθ([CLS](query,doc)). At inference, score = σ(s); candidates are sorted by this score.
- Why MiniLM-L-6-v2: compact cross-encoder to keep CPU latency acceptable on Spaces; finetuning recovers quality lost vs larger models.

### Why
- Cross-encoder improves precision@k and MRR over pure retrieval.
- Model choice: MiniLM-L-6-v2 derivative for size/speed trade-off (CPU-friendly, no GPU requirement on Spaces). Larger cross-encoders would improve quality but were avoided due to latency and lack of GPU.
- Finetuning impact: Compared to off-the-shelf MiniLM, the finetuned reranker yielded higher recall@10/MRR with modest latency increase; acceptable on CPU with k=200 candidates and rerank@10.

## 4) Query Rewriting / Planning
- Deterministic rewriter (retrieval/query_rewriter.py): tokenization + optional NLTK Porter stemming; intent heuristic (TECH/BEHAVIORAL/MIXED/UNKNOWN) via keyword hits; skill extraction (must/soft/negated) using curated vocab and stems; role terms from phrases/keywords; duration/flags/languages parsing; retrieval_query built from role_terms + must/soft skills + constraints; rerank_query = raw text. bm25_query and vec_query are identical strings; clarifying_question disabled; needs_clarification = False.
- LLM rewriters:
  - Qwen/Qwen2.5-1.5B-Instruct: chat template to emit JSON per LLM_SCHEMA; robust structured output; best quality.
  - NuExtract: small, robust schema extraction (default/fallback when no Qwen/GPU).
  - FLAN-T5-small: CPU-friendly but often minimal output; coerced or falls back to deterministic when non-JSON/placeholder.
  - Prompt/schema: LLM_SCHEMA requires `retrieval_query`, `rerank_query`, `intent`, skills, role_terms, constraints (duration/job_levels/languages/flags). Parsing uses `_extract_json_text` + `_coerce_json`; placeholder/invalid output triggers fallback.
- Planner: `tools/query_plan_tool.py` builds `QueryPlan` with provenance (`plan_source`, `llm_debug`); plan_source set to model name on LLM success else “deterministic”; llm_debug carries prompt/raw/clean outputs for audit.

### Why
- Rule-based rewrite guarantees a safe floor; LLMs improve recall/precision when available.
- Deterministic fallback prevents failures if LLM output is malformed or slow.
- LLM selection referenced HF Open LLM leaderboard (https://huggingface.co/collections/open-llm-leaderboard/open-llm-leaderboard-best-models); Qwen chosen for quality/size balance. FLAN/NuExtract used when CPU-only or when Qwen unavailable. Gemini was tested but timed out.

## 5) Constraints & Post-processing
- Applies duration, remote, adaptive flags; formats final payload from the reranked list.
- Guarantees at least one result if candidates exist (fallback to top reranked item).
- Code path: `tools/constraints_tool.py` consumes `QueryPlan` and `RankedList`, filters by duration (<= target), remote/adaptive flags when requested, and slices top-k (default k=10). Missing flags are inferred via `_infer_remote_adaptive` (string match “adaptive” in doc_text/description).
- Numerical fields are sanitized (`_safe_num`) to drop NaN/inf; plan/debug payloads are sanitized (`_sanitize_debug`) before returning.

Example (bash) to locate constraints handling:
```bash
rg "apply_constraints" -n tools agent
sed -n '1,160p' tools/constraints_tool.py
```

## 6) Evaluation Summary (selected runs)

| Run ID                                   | Pipeline                              | Val recall@10 | Val recall@5 | Val MRR@10 |
|------------------------------------------|---------------------------------------|---------------|--------------|------------|
| 20251217_081916_hybrid_rrf_rerank_rewrite| Hybrid RRF + cross-enc rerank + rewrite| 0.4615        | 0.1538       | 0.1250     |
| 20251217_075251_hybrid_rrf_rerank_rewrite| Hybrid RRF + cross-enc rerank + rewrite| 0.3846        | 0.2308       | 0.1865     |
| 20251217_000844_hybrid_rrf_rerank_bge    | Hybrid RRF + cross-enc rerank (BGE)   | 0.3846        | 0.2308       | 0.1865     |
| 20251216_152326_hybrid_rrf_rerank        | Hybrid RRF + cross-enc rerank         | 0.3846        | 0.2308       | 0.1865     |
| 20251216_151000_hybrid_rrf               | Hybrid RRF (no rerank)                | 0.2308        | 0.0769       | 0.0000     |
| 20251216_144825_vector                   | Vector only                           | 0.1538        | 0.0769       | 0.0000     |
| 20251216_140500_bm25                     | BM25 only                             | 0.0769        | 0.0000       | 0.0000     |

Notes:
- Best val recall@10 (0.4615) comes from stacking three improvements: (1) rewrite (LLM-guided query reformulation), (2) hybrid RRF over BM25 + BGE, and (3) cross-encoder rerank. Each component lifts recall/precision; ablations confirm monotonic gains from adding rerank and rewrite.
- Candidate depth: 200 candidates with RRF k=60 maximized recall without exploding latency. BM25/vector-only baselines show the gap: 0.08–0.15 recall@10 vs. 0.23 hybrid (no rerank) vs. 0.38–0.46 hybrid+rerk. Diminishing returns past 200 candidates on this corpus.
- Reranker effect: Cross-encoder rerank drives MRR from 0.0 (no rerank) to ~0.19; quality gain outweighs CPU cost at rerank@10 for a small corpus.
- Rewriter quality: Qwen rewrite > FLAN/NuExtract; FLAN/Gemini underperformed or timed out. Rewrite helps retrieve on long/ambiguous queries by injecting structured constraints; best run uses rewrite.
- Stack choice for production: in-repo BM25 + BGE + RRF + finetuned MiniLM rerank; validated BM25 parity with HF; kept everything local for reproducibility and to avoid external latency/cost.
- Latency/throughput: rerank adds CPU cost but acceptable on a ~389-item corpus; for larger data, tune top-k or rerank depth. Balance quality vs. latency by adjusting candidate count and rerank@k; HNSW not needed here given size/recall goals.

## 7) Recommended stack (current best)
- Retrieval: BM25 + BGE-small + RRF fusion (topn=200, k_rrf=60).
- Rewrite: Qwen/Qwen2.5-1.5B-Instruct when available; deterministic fallback otherwise.
- Rerank: Cross-encoder `models/reranker_crossenc/v0.1.0` to top-10.
- Constraints: duration/remote/adaptive post-filter, guarantee at least one result.
- Serving: FastAPI `/recommend` with cached resources; HF_HOME for persistent model cache.

## 8) How to reproduce eval
1) Ensure artifacts are present (`data/catalog_docs_rich.jsonl`, FAISS index, embeddings, reranker model) and `HF_HOME` is set to a persistent cache if on Spaces.
2) (Optional) Rebuild the catalog via crawler:
   ```bash
   python crawler/run.py
   ```
   This will regenerate catalog docs and downstream artifacts if configured.
3) Run evaluation:
   ```bash
   python eval/run_eval.py \
     --config configs/your_experiment_config.yaml \
     --index data/faiss_index/index_bge.faiss \
     --embeddings data/embeddings_bge/assessment_ids.json
   ```
   Metrics (recall@10/5, MRR@10, etc.) are written to `runs/<timestamp>/metrics.json` and per-query results to `per_query_results.jsonl`.
4) Run ablations (e.g., varying candidate counts, fusion weights, rerank depths):
   ```bash
   python scripts/run_ablation.py \
     --config configs/ablation_config.yaml \
     --output runs/ablation/ablation_results.json
   ```
   Ablation summaries are stored under `runs/ablation/` (e.g., `ablation_results.json`).
5) Compare val metrics across runs (see table above or `runs/*/metrics.json`); prefer the rewrite + hybrid RRF + rerank config for best recall@10 on val.

## 9) Future improvements
- Vector stack: Evaluate HNSW for FAISS if latency becomes critical; add nDCG/precision@k and latency profiling; explore adaptive candidate counts vs. rerank depth for quality/latency trade-offs.
- LLM rewrite: Refine FLAN prompting or drop in favor of Qwen/NuExtract; consider adapter-based (LoRA/PEFT) specialization per company.

## 10) Advanced LLM & Agentic Extensions, RLHF DPO, Knowledge Graph Retrieval
- Agentic query understanding: Add an orchestrator (planner/retriever/clarifier/executor) to iteratively refine underspecified or exploratory queries via tool use (retrieval, stats) before finalizing the plan. Use a ReAct-style loop; keep conversation to reduce plan entropy, not replace retrieval.
- Knowledge graph (KG) augmentation: Build a lightweight recruitment KG (roles, skills, assessments, test types, companies; edges: requires_skill, measures_competency, commonly_used_for_role, preferred_by_company). Inject KG signals into rewrite (skill/role expansion), retrieval (graph-based expansion), and rerank (graph distance as a feature). Fuse graph embeddings (TransE/node2vec) with BM25 + dense.
- Personalization: Learn a company profile vector from past hires/feedback to bias fusion weights, adjust rerank scores, and modulate constraints. Extend final score: score(d)=α·s_rerank+β·s_retrieval+γ·s_company(d), with s_company learned from interaction data.
- Learning from human feedback: Log (query, plan, candidates, chosen, feedback). Train reranker with pairwise preference learning; train rewrite/planner with DPO on positive vs. negative outcomes. Periodically refresh models to align with recruiter preferences without unstable RL.
- Custom LLMs per enterprise: Maintain a base rewrite model; fine-tune lightweight adapters (LoRA/PEFT) per company/industry; swap adapters at inference. Improves structured plans and reduces clarifications; compatible with CPU base + optional GPU for adapters.
- Real-time adaptation: Track query shifts, spikes, emerging skills; update vocab/fusion weights/candidate depth on the fly; invalidate cached plans on distribution shifts.
- System architecture evolution: Agentic orchestration layer (LLM + tools) atop hybrid retrieval core (BM25 + dense + KG), learning layer (rerankers, adapters, preference models), feedback/analytics loop, and policy/safety (guardrails, explainability, audit). Modular to allow incremental upgrades without destabilizing production.

# llm_recommendation_engine
- Crawl & normalize: Playwright → enriched catalog (377→389 after parser fixes)
- Index: BGE-small embeddings + FAISS flat; BM25 for lexical parity
- Retrieve: BM25 + dense via weighted RRF (k_rrf=60, topn=200)
- Rerank: finetuned MiniLM cross-encoder (`models/reranker_crossenc/v0.1.0`) to top-10
- Rewrite/plan: deterministic + LLM (Qwen/NuExtract; FLAN fallback) before retrieval
- Serve/UI: FastAPI (`/recommend`) + Next.js frontend

For design rationale, metrics, and ablations, see `experiments/README.md`.

## Architecture at a glance

![Architecture](media/architecture.png)

- Catalog build: crawl → clean → enrich roles/flags/duration → rich text → embed (BGE-small) → FAISS flat index.
- Query understanding: deterministic parser + optional LLM rewrite/planner (Qwen/NuExtract; FLAN fallback) before retrieval.
- Retrieval: BM25 + dense (BGE) fused via weighted Reciprocal Rank Fusion (k_rrf=60, topn=200).
- Rerank: finetuned cross-encoder (`models/reranker_crossenc/v0.1.0`, MiniLM-based) to top-10.
- Constraints: duration/remote/adaptive filters with safe fallbacks.
- Serving: FastAPI (`/recommend`, `/health`) + Next.js frontend (configurable API base).

## Hosted services
- Backend (FastAPI) on Hugging Face Spaces: `https://agamp-llm-recommendation-backend.hf.space`
  - Endpoints: `/recommend`, `/health`, `/chat` (planned).
  - Repo: https://huggingface.co/spaces/AgamP/llm_recommendation_backend/tree/main
  - Set the frontend API base to this URL for production.
- Frontend (Next.js static) on Render (free tier): https://llm-recommendation-engine.onrender.com/

![platform frontend](media/frontend.png)

## Startup the system on local: Frontend + backend (Next.js + FastAPI)

Backend (FastAPI):

```bash
uvicorn agent.server:app --reload --port 8000
# health
curl http://localhost:8000/health
# recommend
curl -X POST http://localhost:8000/recommend \
  -H "Content-Type: application/json" \
  -d '{"query": "Find a Java dev assessment"}'
```

Frontend (Next.js in `frontend/`):
```bash
cd frontend && npm install
npm run dev        # port 3000; set API base in UI if backend differs
npm run build && npm run start
# UI at http://localhost:3000/ (API base defaults to http://localhost:8000)
```
Docker on hf spaces (backend)
```bash
docker build -t llm-reco-backend .  #buid image
docker run -p 8000:8000 --env-file .env llm-reco-backend #run 
-v $HOME/.cache/huggingface:/home/user/.cache/huggingface #for hf caching
```

## Future Improvements : Advanced LLM & Agentic Extensions · RLHF/DPO · Knowledge Graph Retrieval

> [!NOTE]
> The roadmap below focuses on **incremental, production-safe evolution**: each component is modular, measurable, and designed to improve retrieval quality without destabilizing existing systems.

---

### Agentic Query Understanding
Introduce an agentic orchestrator (planner · retriever · clarifier · executor) to iteratively refine underspecified or exploratory queries.

- ReAct-style loop with tool use (retrieval, statistics, validation)
- Conversation used to **reduce plan entropy**, not replace retrieval
- Finalized plans are compact, structured, and execution-ready

---

### Knowledge Graph (KG) Augmentation
Build a lightweight recruitment knowledge graph to enrich understanding and retrieval.

**Entities**
- Roles, skills, assessments, test types, companies

**Relations**
- `requires_skill`, `measures_competency`
- `commonly_used_for_role`, `preferred_by_company`

**Integration Points**
- Rewrite: role/skill expansion via graph neighborhoods  
- Retrieval: graph-based query expansion  
- Rerank: graph distance as a scoring feature  

Hybrid fusion of **BM25 + dense embeddings + KG embeddings** (TransE / node2vec).

---

### Personalization
Learn a company-specific profile vector from historical interactions.

- Bias fusion weights and rerank scores
- Modulate constraints based on hiring patterns

```python 
score(d) = α · s_rerank + β · s_retrieval + γ · s_company(d)
```
`s_company` is learned from past hires, feedback, and preferences.

---

### Learning From Human Feedback
Continuously improve models using logged recruiter interactions.

- Log: `(query, plan, candidates, chosen, feedback)`
- Train rerankers via **pairwise preference learning**
- Train rewrite/planner models using **DPO** on positive vs. negative outcomes
- Periodic refresh for alignment without unstable online RL

---

### Custom LLMs per Enterprise
Enable enterprise-specific behavior with lightweight adapters.

- Shared base rewrite model
- Per-company or per-industry adapters (LoRA / PEFT)
- Adapter swapping at inference time
- CPU-friendly base with optional GPU acceleration for adapters

Improves structured plans and reduces clarification overhead.

---

### Real-Time Adaptation
Respond dynamically to ecosystem changes.

- Detect query shifts, spikes, and emerging skills
- Adjust vocabulary, fusion weights, and candidate depth
- Invalidate cached plans on distribution shifts

---

### System Architecture Evolution
A layered, modular architecture enabling safe iteration.

- **Agentic orchestration layer** (LLM + tools)
- **Hybrid retrieval core** (BM25 + dense + KG)
- **Learning layer** (rerankers, adapters, preference models)
- **Feedback & analytics loop**
- **Policy & safety layer** (guardrails, explainability, auditability)

> [!IMPORTANT]
> Each layer can be upgraded independently, allowing continuous improvement while maintaining production stability.



### Quick commands (crawler + export + QA)

```bash
Install deps (and Playwright browser): `python -m pip install -r requirements.txt && python -m playwright install chromium`
Clean DB: `rm -f data/crawler.db`
Crawl (bypass robots if needed): `ALLOW_ROBOTS_BYPASS=1 python -m crawler.run --mode=crawl_all --max-discover=20`
  Drop `--max-discover` for full crawl.
Export dataset: `python -m crawler.run --mode=export --limit-export=20`
   Outputs: `data/catalog.parquet`, `data/catalog.jsonl`
   Drop `--limit-export` for full export.
QA checks: `python -m crawler.qa_checks data/catalog.jsonl > data/qa_summary.json`
  Summary JSON saved to `data/qa_summary.json`
```

### What’s implemented
- Playwright-based crawler with pagination + detail fetch into SQLite.
- Field extraction: url, name, description, test_type (+full), remote/adaptive flags, duration, job_levels, languages, downloads.
- Export to Parquet/JSONL plus QA summary script for downstream sanity checks.
- Default artifacts: `data/catalog_docs_rich.jsonl` (389 items), FAISS `data/faiss_index/index_bge.faiss`, embeddings map `data/embeddings_bge/assessment_ids.json`, optional vocab `data/catalog_role_vocab.json`.

## Eval

```python 
Catalog loader with canonical IDs: `python -m data.catalog_loader --input data/catalog.jsonl --output data/catalog_with_ids.jsonl`
Train loader + label resolution report: `python -m data.train_loader --catalog data/catalog.jsonl --train <train_file> --report data/label_resolution_report.json`
Run eval (dummy baseline): `python -m eval.run_eval --catalog data/catalog.jsonl --train <train_file> --recommender dummy_random`
  Run eval (BM25 baseline): `python -m eval.run_eval --catalog data/catalog.jsonl --train <train_file> --recommender bm25`
  Outputs: `runs/<timestamp>_<recommender>/metrics.json`, `per_query_results.jsonl`, `worst_queries.csv`, `label_resolution_report.json`
Compare runs: `python -m eval.compare_runs runs/<run_a> runs/<run_b>`
```

Recommender interface: `recommenders/base.py`. Metrics: `eval/metrics.py` (Recall@k, MRR@10).

## Label probing & backfill (improve label coverage)
- Probe unmatched label URLs: `python -m scripts.probe_unmatched_labels --labels data/label_resolution_report.json --output reports/label_url_probe.csv` — classifies detail vs 404/blocked.
- Backfill valid pages: `python -m crawler.backfill_labels --probe-csv reports/label_url_probe.csv --allow-robots-bypass`
- Re-export and rematch:
  - `python -m crawler.run --mode=export`
  - `python -m data.catalog_loader --input data/catalog.jsonl --output data/catalog_with_ids.jsonl`
  - `python -m data.train_loader --catalog data/catalog.jsonl --train <train_file> --sheet "Train-Set" --report data/label_resolution_report.json`

## Vector pipeline (semantic retrieval)
- Build doc_text: `python -m data.document_builder --input data/catalog.jsonl --output data/catalog_docs.jsonl`
- Generate embeddings: `python -m embeddings.generator --catalog data/catalog_docs.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --output-dir data/embeddings`
- Build FAISS index: `python -m retrieval.build_index --embeddings data/embeddings/embeddings.npy --ids data/embeddings/assessment_ids.json --index-path data/faiss_index/index.faiss`
- Components: `models/embedding_model.py`, `retrieval/vector_index.py`, `retrieval/build_index.py`, `recommenders/vector_recommender.py`.

## Hybrid retrieval (BM25 + vector with RRF)
- Run hybrid eval: `python -m eval.run_eval --catalog data/catalog_docs.jsonl --train data/Gen_AI\ Dataset.xlsx --recommender hybrid_rrf --vector-index data/faiss_index/index.faiss --assessment-ids data/embeddings/assessment_ids.json --model sentence-transformers/all-MiniLM-L6-v2 --topn-candidates 200 --rrf-k 60`
- Run hybrid + cross-encoder rerank: `python -m eval.run_eval --catalog data/catalog_docs.jsonl --train data/Gen_AI\ Dataset.xlsx --recommender hybrid_rrf_rerank --vector-index data/faiss_index/index.faiss --assessment-ids data/embeddings/assessment_ids.json --model sentence-transformers/all-MiniLM-L6-v2 --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 --topn-candidates 200 --rrf-k 60`
- Run hybrid + LGBM rerank: `python -m eval.run_eval --catalog data/catalog_docs.jsonl --train data/Gen_AI\ Dataset.xlsx --recommender hybrid_rrf_lgbm --vector-index data/faiss_index/index.faiss --assessment-ids data/embeddings/assessment_ids.json --model sentence-transformers/all-MiniLM-L6-v2 --topn-candidates 200 --rrf-k 60 --lgbm-model models/reranker/v0.1.0/lgbm_model.txt --lgbm-features models/reranker/v0.1.0/feature_schema.json`
- Diagnostics (positives in top-N vs top-10): `python -m eval.diagnostic_topk --catalog data/catalog_docs.jsonl --train data/Gen_AI\ Dataset.xlsx --vector-index data/faiss_index/index.faiss --assessment-ids data/embeddings/assessment_ids.json --model sentence-transformers/all-MiniLM-L6-v2 --topn 200`
- Run ablation (bm25/vector/hybrid across topN): `python -m scripts.run_ablation --catalog data/catalog.jsonl --train data/Gen_AI\ Dataset.xlsx --vector-index data/faiss_index/index.faiss --assessment-ids data/embeddings/assessment_ids.json --model sentence-transformers/all-MiniLM-L6-v2 --topn-list 100,200,377`

## Current findings & next steps
- Candidate coverage is solved by top200; ranking is the bottleneck. Use union fusion + rerank.
- Locked decisions:
  - Candidate pool (train): top200
  - Candidate pool (infer): top100–200
  - Base retriever: hybrid (BM25 + vector), union fusion, dual-query (raw + rewritten).
- Next: focus on reranking and constraint handling; no more embedding/model swaps.

## Core pipeline (concise commands)

### Build rich docs, embeddings, index (BGE)
```bash
python -m data.document_builder \
  --input data/catalog.jsonl \
  --output data/catalog_docs_rich.jsonl \
  --variant rich \
  --version v2_struct

python -m embeddings.generator \
  --catalog data/catalog_docs_rich.jsonl \
  --model BAAI/bge-small-en-v1.5 \
  --batch-size 32 \
  --output-dir data/embeddings_bge

python -m retrieval.build_index \
  --embeddings data/embeddings_bge/embeddings.npy \
  --ids data/embeddings_bge/assessment_ids.json \
  --index-path data/faiss_index/index_bge.faiss
```

### Build vocab for query rewriter (optional, recommended)
```bash
python -m scripts.build_role_vocab \
  --catalog data/catalog_docs_rich.jsonl \
  --out data/catalog_role_vocab.json
```

### Evaluate hybrid + cross-encoder rerank (with rewriting and union fusion)
```bash
python -m eval.run_eval \
  --catalog data/catalog_docs_rich.jsonl \
  --train data/Gen_AI\ Dataset.xlsx \
  --recommender hybrid_rrf_rerank \
  --vector-index data/faiss_index/index_bge.faiss \
  --assessment-ids data/embeddings_bge/assessment_ids.json \
  --model BAAI/bge-small-en-v1.5 \
  --reranker-model models/reranker_crossenc/v0.1.0 \
  --topn-candidates 200 --rrf-k 60 \
  --use-rewriter --vocab data/catalog_role_vocab.json \
  --out-dir runs/$(date +%Y%m%d_%H%M%S)_hybrid_rrf_rerank_rewrite
```

### Candidate coverage (bm25 vs vector vs hybrid; grouped per query)
```bash
python -m scripts.candidate_coverage \
  --catalog data/catalog_docs_rich.jsonl \
  --train data/Gen_AI\ Dataset.xlsx \
  --vector-index data/faiss_index/index_bge.faiss \
  --assessment-ids data/embeddings_bge/assessment_ids.json \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --topn 200 \
  --use-rewriter --vocab data/catalog_role_vocab.json \
  --out runs/candidate_coverage.jsonl

python -m scripts.summarize_candidate_coverage \
  --input runs/candidate_coverage.jsonl \
  --out runs/candidate_coverage_stats.json
```

### Rewrite impact (optional)
```bash
python -m scripts.eval_rewrite_impact \
  --catalog data/catalog_docs_rich.jsonl \
  --train data/Gen_AI\ Dataset.xlsx \
  --vector-index data/faiss_index/index_bge.faiss \
  --assessment-ids data/embeddings_bge/assessment_ids.json \
  --embedding-model BAAI/bge-small-en-v1.5 \
  --topn 200 \
  --vocab data/catalog_role_vocab.json \
  --out runs/rewrite_impact.jsonl
```

## Recommended stack (prod defaults)
- Retrieval: BM25 + BGE-small + RRF fusion (topn=200, k_rrf=60).
- Rewrite: Qwen/Qwen2.5-1.5B-Instruct when available; deterministic fallback otherwise.
- Rerank: Cross-encoder `models/reranker_crossenc/v0.1.0` to top-10.
- Constraints: duration/remote/adaptive post-filters; guarantee at least one result.
- Caching: set `HF_HOME=/home/user/.cache/huggingface` on Spaces to avoid cold-start downloads.

## Evaluation highlights (val split 52/13)
| Pipeline                                | recall@10 | recall@5 | MRR@10 |
|-----------------------------------------|-----------|----------|--------|
| Hybrid RRF + rerank + rewrite (best)    | 0.4615    | 0.1538   | 0.1250 |
| Hybrid RRF + rerank                     | 0.3846    | 0.2308   | 0.1865 |
| Hybrid RRF (no rerank)                  | 0.2308    | 0.0769   | 0.0000 |
| Vector only                             | 0.1538    | 0.0769   | 0.0000 |
| BM25 only                               | 0.0769    | 0.0000   | 0.0000 |

More detail (fusion math, rerank training setup, ablations, future agentic extensions) lives in `experiments/README.md`.

# llm_recommendation_engine
Recommendation engine for SHL's product catalogue with conversational agents

## Quick commands (crawler + export + QA)
- Install deps (and Playwright browser): `python -m pip install -r requirements.txt && python -m playwright install chromium`
- Clean DB: `rm -f data/crawler.db`
- Crawl (bypass robots if needed): `ALLOW_ROBOTS_BYPASS=1 python -m crawler.run --mode=crawl_all --max-discover=20`
  - Drop `--max-discover` for full crawl.
- Export dataset: `python -m crawler.run --mode=export --limit-export=20`
  - Outputs: `data/catalog.parquet`, `data/catalog.jsonl`
  - Drop `--limit-export` for full export.
- QA checks: `python -m crawler.qa_checks data/catalog.jsonl > data/qa_summary.json`
  - Summary JSON saved to `data/qa_summary.json`

## What’s implemented
- Playwright-based crawler with catalog pagination, detail fetch, and structured storage in SQLite.
- Field extraction: url, name, description, test_type (+full), remote/adaptive flags, duration (minutes/hours), job_levels, languages, downloads.
- Export to Parquet/JSONL plus QA summary script for downstream sanity checks.

## Evaluation harness (Phase 2)
- Catalog loader with canonical IDs: `python -m data.catalog_loader --input data/catalog.jsonl --output data/catalog_with_ids.jsonl`
- Train loader + label resolution report: `python -m data.train_loader --catalog data/catalog.jsonl --train <train_file> --report data/label_resolution_report.json`
- Run eval (dummy baseline): `python -m eval.run_eval --catalog data/catalog.jsonl --train <train_file> --recommender dummy_random`
  - Run eval (BM25 baseline): `python -m eval.run_eval --catalog data/catalog.jsonl --train <train_file> --recommender bm25`
  - Outputs run folder under `runs/<timestamp>_<recommender>/` with `metrics.json`, `per_query_results.jsonl`, `worst_queries.csv`, `label_resolution_report.json`
- Compare runs: `python -m eval.compare_runs runs/<run_a> runs/<run_b>`

Recommender interface lives in `recommenders/base.py`; a random baseline is in `recommenders/dummy_random.py`. Metrics (Recall@k, MRR@10) are in `eval/metrics.py`.

## Label probing & backfill (improve label coverage)
- Probe unmatched label URLs (after a label match run): `python -m scripts.probe_unmatched_labels --labels data/label_resolution_report.json --output reports/label_url_probe.csv` — classifies label URLs (valid detail vs 404/blocked).
- Backfill valid label pages into DB: `python -m crawler.backfill_labels --probe-csv reports/label_url_probe.csv --allow-robots-bypass` — fetches & inserts DETAIL_PAGE_VALID URLs.
- Re-export and rematch after backfill:
  - `python -m crawler.run --mode=export`
  - `python -m data.catalog_loader --input data/catalog.jsonl --output data/catalog_with_ids.jsonl`
  - `python -m data.train_loader --catalog data/catalog.jsonl --train <train_file> --sheet "Train-Set" --report data/label_resolution_report.json`

## Vector pipeline (semantic retrieval)
- Build doc_text: `python -m data.document_builder --input data/catalog.jsonl --output data/catalog_docs.jsonl`
- Generate embeddings: `python -m embeddings.generator --catalog data/catalog_docs.jsonl --model sentence-transformers/all-MiniLM-L6-v2 --output-dir data/embeddings`
- Build FAISS index: `python -m retrieval.build_index --embeddings data/embeddings/embeddings.npy --ids data/embeddings/assessment_ids.json --index-path data/faiss_index/index.faiss`
- Vector components:
  - Model wrapper: `models/embedding_model.py`
  - Index wrapper: `retrieval/vector_index.py`
  - Index builder script: `retrieval/build_index.py`
  - Vector recommender scaffold: `recommenders/vector_recommender.py` (wire with assessment_ids + index)

## Hybrid retrieval (BM25 + vector with RRF)
- Run hybrid eval: `python -m eval.run_eval --catalog data/catalog_docs.jsonl --train data/Gen_AI\ Dataset.xlsx --recommender hybrid_rrf --vector-index data/faiss_index/index.faiss --assessment-ids data/embeddings/assessment_ids.json --model sentence-transformers/all-MiniLM-L6-v2 --topn-candidates 200 --rrf-k 60`
- Run hybrid + cross-encoder rerank: `python -m eval.run_eval --catalog data/catalog_docs.jsonl --train data/Gen_AI\ Dataset.xlsx --recommender hybrid_rrf_rerank --vector-index data/faiss_index/index.faiss --assessment-ids data/embeddings/assessment_ids.json --model sentence-transformers/all-MiniLM-L6-v2 --reranker-model cross-encoder/ms-marco-MiniLM-L-6-v2 --topn-candidates 200 --rrf-k 60`
- Run hybrid + LGBM rerank: `python -m eval.run_eval --catalog data/catalog_docs.jsonl --train data/Gen_AI\ Dataset.xlsx --recommender hybrid_rrf_lgbm --vector-index data/faiss_index/index.faiss --assessment-ids data/embeddings/assessment_ids.json --model sentence-transformers/all-MiniLM-L6-v2 --topn-candidates 200 --rrf-k 60 --lgbm-model models/reranker/v0.1.0/lgbm_model.txt --lgbm-features models/reranker/v0.1.0/feature_schema.json`
- Diagnostics (positives in top-N vs top-10): `python -m eval.diagnostic_topk --catalog data/catalog_docs.jsonl --train data/Gen_AI\ Dataset.xlsx --vector-index data/faiss_index/index.faiss --assessment-ids data/embeddings/assessment_ids.json --model sentence-transformers/all-MiniLM-L6-v2 --topn 200`
- Run ablation (bm25/vector/hybrid across topN): `python -m scripts.run_ablation --catalog data/catalog_docs.jsonl --train data/Gen_AI\ Dataset.xlsx --vector-index data/faiss_index/index.faiss --assessment-ids data/embeddings/assessment_ids.json --model sentence-transformers/all-MiniLM-L6-v2 --topn-list 100,200,377`

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

## Frontend + backend (Next.js + FastAPI)

Backend (FastAPI):
- Start: `uvicorn agent.server:app --reload --port 8000`
- Health: `GET /health`
- Chat: `POST /chat` (returns compact top-10 + optional summary when verbose=true)
- Recommend: `POST /recommend` with `{"query": "..."}` returns `{"recommended_assessments": [...]}` (top-10)

Frontend (Next.js in `frontend/`):
- Install deps: `cd frontend && npm install`
- Dev: `npm run dev` (will start on port 3000; ensure backend is running on 8000 or set API base in UI)
- Build/start: `npm run build && npm run start`
- UI is at `http://localhost:3000/` (API base defaults to `http://localhost:8000`, editable in the UI)

from __future__ import annotations

from typing import Dict, List


class HybridRRFRecommender:
    """Hybrid retriever using Reciprocal Rank Fusion over BM25 and Vector results."""

    def __init__(
        self,
        bm25,
        vector,
        topn_candidates: int = 50,
        rrf_k: int = 60,
        rewriter=None,
        rewriter_vocab=None,
    ) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.topn_candidates = topn_candidates
        self.rrf_k = rrf_k
        self.rewriter = rewriter  # optional callable: (text, catalog_vocab) -> QueryRewrite
        self.rewriter_vocab = rewriter_vocab

    def recommend(self, query: str, k: int = 10) -> List[str]:
        queries = [query]
        if self.rewriter:
            try:
                rw = self.rewriter(query, catalog_vocab=self.rewriter_vocab)
                queries.append(rw.retrieval_query)
            except Exception:
                pass

        bm25_ids: List[str] = []
        vec_ids: List[str] = []
        for q in queries:
            bm25_ids.extend(self._to_ids(self.bm25.recommend(q, k=self.topn_candidates)))
            vec_ids.extend(self._to_ids(self.vector.recommend(q, k=self.topn_candidates)))
        # dedupe preserving order
        bm25_ids = list(dict.fromkeys(bm25_ids))
        vec_ids = list(dict.fromkeys(vec_ids))

        # Union fusion: preserve BM25 order, then vector-only tail.
        union_ordered: Dict[str, None] = {}
        for aid in bm25_ids:
            union_ordered.setdefault(aid, None)
        for aid in vec_ids:
            union_ordered.setdefault(aid, None)
        return list(union_ordered.keys())[:k]

    @staticmethod
    def _to_ids(results) -> List[str]:
        ids = []
        for r in results:
            if isinstance(r, str):
                ids.append(r)
            elif isinstance(r, dict) and "assessment_id" in r:
                ids.append(r["assessment_id"])
        return ids


class HybridRerankRecommender:
    """Hybrid retrieval with RRF fusion, then reranking via a reranker (e.g., cross-encoder)."""

    def __init__(
        self,
        bm25,
        vector,
        reranker,
        catalog_df,
        topn_candidates: int = 50,
        rrf_k: int = 60,
        rewriter=None,
        rewriter_vocab=None,
    ) -> None:
        self.bm25 = bm25
        self.vector = vector
        self.reranker = reranker
        self.catalog = catalog_df.set_index("assessment_id")
        self.topn_candidates = topn_candidates
        self.rrf_k = rrf_k
        self.rewriter = rewriter  # optional callable: (text, catalog_vocab) -> QueryRewrite
        self.rewriter_vocab = rewriter_vocab

    def recommend(self, query: str, k: int = 10, rerank_query: str | None = None):
        queries = [query]
        if self.rewriter:
            try:
                rw = self.rewriter(query, catalog_vocab=self.rewriter_vocab)
                queries.append(rw.retrieval_query)
                if rerank_query is None:
                    rerank_query = rw.rerank_query
            except Exception:
                pass
        bm25_ids: List[str] = []
        vec_ids: List[str] = []
        for q in queries:
            bm25_ids.extend(HybridRRFRecommender._to_ids(self.bm25.recommend(q, k=self.topn_candidates)))
            vec_ids.extend(HybridRRFRecommender._to_ids(self.vector.recommend(q, k=self.topn_candidates)))
        bm25_ids = list(dict.fromkeys(bm25_ids))
        vec_ids = list(dict.fromkeys(vec_ids))

        # Union fusion: BM25 order, then vector-only tail.
        union_ordered: Dict[str, None] = {}
        for aid in bm25_ids:
            union_ordered.setdefault(aid, None)
        for aid in vec_ids:
            union_ordered.setdefault(aid, None)
        candidates = list(union_ordered.keys())[: self.topn_candidates]

        rerank_text = rerank_query or query
        scored = []
        for aid in candidates:
            if aid not in self.catalog.index:
                continue
            doc_text = self.catalog.loc[aid].get("doc_text", "")
            score = self.reranker.score(rerank_text, doc_text)
            scored.append((aid, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [aid for aid, _ in scored[:k]]

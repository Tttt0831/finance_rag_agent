"""
services/hybrid_retriever.py - 混合检索（BM25 + 稠密向量 → RRF 融合 → 可选 rerank → 父子块展开）

对照《Agent 开发知识库》模块 03「召回段三板斧」：
  1. BM25 补稠密检索对专有名词/型号/错误码的召回短板；
  2. RRF 只用排名不用分数，避开两路分数量纲不可比问题；
  3. 召回放宽到 top-N，再用 cross-encoder 精排（可选，默认关）；
  4. 命中子块后按 parent_id 去重取回父块（small-to-big，可选）。

BM25 语料懒加载：首次查询时从向量库全量拉取，文档版本变化时自动重建。
"""

from typing import List, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from config import settings
from services.vector_store import vector_store_service
from middleware.trace import get_trace_thread_id, trace_store
from utils.logger import get_logger

logger = get_logger(__name__)


def _tokenize(text: str) -> List[str]:
    """中文分词（jieba），未安装时退化为字符级切分。"""
    try:
        import jieba
        return list(jieba.lcut(text))
    except ImportError:
        return list(text)


class HybridRetriever(BaseRetriever):
    """BM25 + 稠密混合检索器（LangChain Retriever 接口）。"""

    def __init__(self):
        super().__init__()
        self._bm25 = None
        self._corpus: List[Document] = []
        self._tokenized_corpus: List[List[str]] = []
        self._built_version: int = -1
        self._reranker = None

    # ── BM25 语料维护 ───────────────────────────────────────

    def _rebuild_if_needed(self) -> None:
        """文档版本变化时重建 BM25 语料。"""
        if self._built_version == vector_store_service.version:
            return
        docs = vector_store_service.get_all_documents()
        self._corpus = docs
        self._tokenized_corpus = [_tokenize(d.page_content) for d in docs]
        if docs:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized_corpus)
        else:
            self._bm25 = None
        self._built_version = vector_store_service.version
        logger.info(f"[混合检索] BM25 语料已构建: {len(docs)} 块")

    # ── 三路召回 ────────────────────────────────────────────

    def _dense_search(self, query: str, k: int) -> List[Document]:
        return vector_store_service.similarity_search(query, k=k)

    def _sparse_search(self, query: str, k: int) -> List[Document]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        idxs = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self._corpus[i] for i in idxs]

    @staticmethod
    def _rrf(lists: List[List[Document]]) -> List[tuple]:
        """Reciprocal Rank Fusion：只吃排名不吃分数。返回 [(doc, fused_score)]。"""
        fused: dict = {}
        docmap: dict = {}
        for lst in lists:
            for rank, doc in enumerate(lst, start=1):
                key = doc.page_content   # 以正文去重，dense/sparse 两路同源
                docmap.setdefault(key, doc)
                fused[key] = fused.get(key, 0.0) + 1.0 / (settings.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda kv: -kv[1])
        return [(docmap[k], s) for k, s in ranked]

    # ── 可选 rerank（P3，默认关）─────────────────────────────

    def _rerank(self, query: str, docs: List[Document]) -> List[Document]:
        try:
            from sentence_transformers import CrossEncoder
            if self._reranker is None:
                logger.info(f"[rerank] 加载模型 {settings.rerank_model} ...")
                self._reranker = CrossEncoder(settings.rerank_model)
            pairs = [(query, d.page_content) for d in docs]
            scores = self._reranker.predict(pairs)
            return [d for _, d in sorted(zip(scores, docs), key=lambda x: -x[0])]
        except Exception as e:
            logger.warning(f"[rerank] 不可用，回退 RRF 顺序: {e}")
            return docs

    # ── 可选父子块展开（P3，默认关）─────────────────────────

    def _expand_parents(self, candidates: List[Document], k: int) -> List[Document]:
        if not settings.enable_parent_child:
            return candidates[:k]
        seen, order = set(), []
        for d in candidates:
            pid = d.metadata.get("parent_id")
            if pid and pid not in seen:
                seen.add(pid)
                order.append(pid)
        if not order:
            return candidates[:k]
        pmap = {p.metadata.get("parent_id"): p for p in vector_store_service.get_parents(order)}
        return [pmap[pid] for pid in order if pid in pmap][:k]

    # ── 主入口 ──────────────────────────────────────────────

    def search(self, query: str, k: Optional[int] = None,
               use_hybrid: Optional[bool] = None) -> List[Document]:
        """混合检索。use_hybrid=False 时退化为纯稠密（供评测 A/B）。"""
        k = k or settings.retriever_k
        use_hybrid = settings.enable_hybrid if use_hybrid is None else use_hybrid

        self._rebuild_if_needed()
        dense = self._dense_search(query, settings.hybrid_k)
        if use_hybrid:
            sparse = self._sparse_search(query, settings.hybrid_k)
            fused = self._rrf([dense, sparse])
        else:
            fused = self._rrf([dense])

        # 记录检索命中（doc_id/来源/RRF 分数），供 trace 归因
        self._record(query, fused, use_hybrid)

        candidates = [d for d, _ in fused]
        if settings.enable_rerank and len(candidates) > 1:
            candidates = (
                self._rerank(query, candidates[: settings.rerank_top_k])
                + candidates[settings.rerank_top_k :]
            )
        return self._expand_parents(candidates, k)

    def _record(self, query: str, fused: List[tuple], use_hybrid: bool) -> None:
        thread = get_trace_thread_id()
        if not thread:
            return
        payload = [
            {"metadata": d.metadata, "score": round(s, 6)}
            for d, s in fused[: settings.hybrid_k]
        ]
        trace_store.record_retrieval(
            thread, query, payload, method="hybrid" if use_hybrid else "dense"
        )

    # LangChain Retriever 接口
    def _get_relevant_documents(
        self, query: str, *, run_manager=None
    ) -> List[Document]:
        return self.search(query)


# 全局单例（rag_service / eval 共用）
hybrid_retriever = HybridRetriever()

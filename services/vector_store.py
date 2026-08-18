"""
services/vector_store.py - 向量存储服务
负责：文档加载 → 文本分割 → Embedding → 存入/读取 Chroma 向量库
"""

import hashlib
import os
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    CSVLoader,
    TextLoader,
    UnstructuredExcelLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import settings
from utils.logger import get_logger
from utils.path import get_file_md5

logger = get_logger(__name__)

# MD5 去重记录文件
_MD5_RECORD_FILE = os.path.join(settings.upload_dir, ".indexed_md5.txt")


def _load_indexed_md5() -> set:
    """读取已入库的文件 MD5 集合。"""
    if not os.path.exists(_MD5_RECORD_FILE):
        return set()
    with open(_MD5_RECORD_FILE, "r") as f:
        return set(line.strip() for line in f if line.strip())


def _save_md5(md5: str):
    """将新文件 MD5 追加写入记录。"""
    with open(_MD5_RECORD_FILE, "a") as f:
        f.write(md5 + "\n")


class VectorStoreService:
    """向量存储服务（单例模式）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info("初始化向量存储服务...")

        self._embeddings = HuggingFaceEmbeddings(
            model_name=settings.embedding_model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        self._vectorstore = Chroma(
            persist_directory=settings.chroma_persist_dir,
            embedding_function=self._embeddings,
            collection_name="finance_knowledge",
        )
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
        )
        self._indexed_md5 = _load_indexed_md5()
        # 文档变更版本号：入库后 +1，混合检索据此重建 BM25 语料
        self._version = 0
        # 父子块模式下的父块集合（懒加载，见 _parents_store）
        self._parents_store = None
        self._initialized = True
        logger.info("向量存储服务初始化完成。")

    # ── 公开方法 ─────────────────────────────────────────────

    def get_retriever(self):
        """返回 LangChain Retriever，供 RAG 和 Agent 使用。"""
        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": settings.retriever_k},
        )

    def add_file(self, file_path: str) -> dict:
        """
        将文件入库。自动根据扩展名选择 Loader，并做 MD5 去重。

        当 settings.enable_parent_child 为 True 时走父子块双写，
        否则走扁平切分（默认，向后兼容）。

        Returns:
            {"status": "success"|"skipped", "chunks": int, "md5": str}
        """
        with open(file_path, "rb") as f:
            file_bytes = f.read()
        md5 = get_file_md5(file_bytes)

        if md5 in self._indexed_md5:
            logger.info(f"[跳过] 文件已入库: {os.path.basename(file_path)} ({md5[:8]}...)")
            return {"status": "skipped", "chunks": 0, "md5": md5}

        source = os.path.basename(file_path)
        docs = self._load_file(file_path)
        if settings.enable_parent_child:
            n = self._add_small2big(docs, source, md5)
        else:
            chunks = self._splitter.split_documents(docs)
            for chunk in chunks:
                chunk.metadata["source_file"] = source
                chunk.metadata["md5"] = md5
            self._vectorstore.add_documents(chunks)
            n = len(chunks)

        _save_md5(md5)
        self._indexed_md5.add(md5)
        self._version += 1

        logger.info(f"[入库成功] {source} → {n} 个块")
        return {"status": "success", "chunks": n, "md5": md5}

    def add_texts(self, texts: List[str], metadatas: List[dict] = None) -> int:
        """直接将文本列表入库（用于内置知识预加载）。"""
        docs = [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metadatas or [{}] * len(texts))
        ]
        if settings.enable_parent_child:
            n = self._add_small2big(docs, "内置知识", None)
        else:
            chunks = self._splitter.split_documents(docs)
            self._vectorstore.add_documents(chunks)
            n = len(chunks)
        self._version += 1
        logger.info(f"[文本入库] {len(texts)} 条文本 → {n} 个块")
        return n

    # ── 检索辅助（供混合检索 / 父子块展开使用）─────────────

    @property
    def version(self) -> int:
        """文档变更版本号，混合检索据此判断是否重建 BM25。"""
        return self._version

    def get_all_documents(self) -> List[Document]:
        """读取主集合全部文档（供 BM25 语料构建）。"""
        data = self._vectorstore.get()
        ids = data.get("ids", [])
        texts = data.get("documents", [])
        metas = data.get("metadatas", []) or [{}] * len(texts)
        return [
            Document(page_content=t, metadata={**(m or {}), "id": i})
            for i, t, m in zip(ids, texts, metas)
        ]

    def get_parents(self, parent_ids: List[str]) -> List[Document]:
        """按 parent_id 直接取回父块（无需向量检索）。"""
        if not parent_ids:
            return []
        store = self._get_parents_store()
        if store is None:
            return []
        data = store.get(ids=list(parent_ids))
        texts = data.get("documents", [])
        metas = data.get("metadatas", []) or [{}] * len(texts)
        return [
            Document(page_content=t, metadata=m or {})
            for t, m in zip(texts, metas)
        ]

    def _get_parents_store(self) -> "Chroma":
        """父块集合（懒加载）。"""
        if self._parents_store is None:
            self._parents_store = Chroma(
                persist_directory=settings.chroma_persist_dir,
                embedding_function=self._embeddings,
                collection_name="finance_knowledge_parents",
            )
        return self._parents_store

    def _add_small2big(self, docs: List[Document], source: str, md5: str) -> int:
        """
        父子块（small-to-big）双写：
        - 父块（大）入父集合，带 parent_id；
        - 子块（小）入主集合，带 parent_id metadata 指向父块。
        标题（此处以来源文件名近似）随子块复制，缓解语义漂移。
        """
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.parent_chunk_size, chunk_overlap=40,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.child_chunk_size, chunk_overlap=32,
            separators=["\n\n", "\n", "。", "；", "！", "？", "，", " ", ""],
        )
        parents = parent_splitter.split_documents(docs)
        parent_docs, child_docs = [], []
        for p in parents:
            pid = hashlib.md5(f"{source}:{p.page_content}".encode("utf-8")).hexdigest()
            parent_docs.append(Document(
                page_content=p.page_content,
                metadata={"parent_id": pid, "source_file": source, "md5": md5},
            ))
            for c in child_splitter.split_text(p.page_content):
                child_docs.append(Document(
                    page_content=f"{source}：{c}",   # 标题随块复制
                    metadata={"parent_id": pid, "source_file": source, "md5": md5},
                ))
        self._get_parents_store().add_documents(parent_docs)
        self._vectorstore.add_documents(child_docs)
        logger.info(f"[父子块入库] {source} → {len(parents)} 父块 / {len(child_docs)} 子块")
        return len(child_docs)

    def similarity_search(self, query: str, k: int = None) -> List[Document]:
        """直接执行相似度搜索，返回 Document 列表。"""
        return self._vectorstore.similarity_search(query, k=k or settings.retriever_k)

    # ── 私有方法 ─────────────────────────────────────────────

    def _load_file(self, file_path: str) -> List[Document]:
        """根据文件扩展名自动选择 Loader。"""
        ext = os.path.splitext(file_path)[1].lower()
        loader_map = {
            ".pdf": lambda p: PyPDFLoader(p),
            ".csv": lambda p: CSVLoader(p, encoding="utf-8"),
            ".txt": lambda p: TextLoader(p, encoding="utf-8"),
            ".md": lambda p: TextLoader(p, encoding="utf-8"),
            ".xlsx": lambda p: UnstructuredExcelLoader(p),
            ".xls": lambda p: UnstructuredExcelLoader(p),
        }
        loader_fn = loader_map.get(ext)
        if loader_fn is None:
            raise ValueError(f"不支持的文件类型: {ext}，支持: {list(loader_map.keys())}")
        return loader_fn(file_path).load()


# 全局单例
vector_store_service = VectorStoreService()

"""
services/rag_service.py - RAG 问答服务
基于「查询改写 → 混合检索 → LLM 回答」金融知识库中的问题
"""

import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from config import settings
from services.hybrid_retriever import hybrid_retriever
from utils.logger import get_logger

logger = get_logger(__name__)

# ── RAG Prompt 模板 ──────────────────────────────────────────
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一位专业的金融分析师助手。请根据以下【知识库内容】回答用户问题。

回答要求：
1. 优先使用知识库中的信息，引用时请说明"根据知识库..."
2. 如知识库中无相关信息，可结合自身知识回答，但需注明"以下为通用金融知识"
3. 涉及投资建议时，必须在末尾加免责声明："以上内容仅供参考，不构成投资建议，投资有风险，入市须谨慎。"
4. 回答要专业、简洁、有条理，可使用列表或小标题

【知识库内容】
{context}
""",
    ),
    ("human", "{question}"),
])

# ── 查询改写 Prompt（P2，默认关）────────────────────────────
_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """把用户的金融提问改写为可独立检索的查询。规则：
1. 补全口语中的指代与省略，使查询自包含；
2. 若包含多个独立意图，拆成多条（最多 3 条）；
3. 保留原文中的专有名词、代码、型号，不要意译；
4. 只输出 JSON 数组，不要输出任何其他内容。""",
    ),
    ("human", "{question}"),
])


def _format_docs(docs) -> str:
    """将检索到的文档列表格式化为字符串，附带来源信息。"""
    if not docs:
        return "（知识库中未检索到相关内容）"
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source_file", "内置知识")
        parts.append(f"[来源{i}: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


class RagService:
    """RAG 问答服务（单例）"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info("初始化 RAG 问答服务...")

        self._llm = ChatOpenAI(
            model=settings.model_name,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        self._retriever = hybrid_retriever
        self._chain = _RAG_PROMPT | self._llm | StrOutputParser()
        self._initialized = True
        logger.info("RAG 问答服务初始化完成。")

    # ── 查询改写（P2）──────────────────────────────────────

    def _rewrite(self, question: str) -> list:
        """口语/指代补全 + 多意图分解，失败时回退原 query。"""
        if not settings.enable_query_rewrite:
            return [question]
        try:
            raw = (_REWRITE_PROMPT | self._llm | StrOutputParser()).invoke(
                {"question": question}
            )
            queries = json.loads(raw.strip())
            if isinstance(queries, list) and queries:
                return [str(q) for q in queries][:3]
        except Exception as e:
            logger.warning(f"查询改写失败，回退原 query: {e}")
        return [question]

    def _retrieve(self, question: str) -> list:
        """改写后逐条检索，按正文去重合并。"""
        docs = []
        for q in self._rewrite(question):
            docs.extend(self._retriever.search(q, k=settings.retriever_k))
        seen, out = set(), []
        for d in docs:
            if d.page_content not in seen:
                seen.add(d.page_content)
                out.append(d)
        return out[: settings.retriever_k * 2]

    # ── 问答 ───────────────────────────────────────────────

    def query(self, question: str, chat_history: list = None) -> str:
        """单次问答（供 Agent 工具调用）。"""
        question = str(question)
        try:
            docs = self._retrieve(question)
            result = self._chain.invoke(
                {"context": _format_docs(docs), "question": question}
            )
            logger.debug(f"RAG 回答: {str(result)[:100]}...")
            return str(result)
        except Exception as e:
            logger.error(f"RAG invoke 失败: {e}")
            # 降级：直接返回检索到的文档原文
            docs = self._retriever.search(question, k=3)
            if docs:
                parts = [f"[来源:{d.metadata.get('source_file','知识库')}] {d.page_content}" for d in docs[:3]]
                return "（RAG 生成失败，返回原始检索结果）\n\n" + "\n\n---\n\n".join(parts)
            return f"知识库检索失败: {str(e)[:100]}"

    def stream_query(self, question: str):
        """流式问答，返回生成器（供前端实时展示）。"""
        docs = self._retrieve(question)
        for chunk in self._chain.stream(
            {"context": _format_docs(docs), "question": question}
        ):
            yield chunk


# 全局单例
rag_service = RagService()

"""
services/rag_service.py - RAG 问答服务
基于向量检索 + LLM 回答金融知识库中的问题
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from config import settings
from services.vector_store import vector_store_service
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
        self._retriever = vector_store_service.get_retriever()
        self._chain = (
            {
                "context": self._retriever | _format_docs,
                "question": RunnablePassthrough(),
            }
            | _RAG_PROMPT
            | self._llm
            | StrOutputParser()
        )
        self._initialized = True
        logger.info("RAG 问答服务初始化完成。")

    def query(self, question: str, chat_history: list = None) -> str:
        """
        单次问答（供 Agent 工具调用）。
        """
        question = str(question)
        try:
            result = self._chain.invoke(question)
            logger.debug(f"RAG 回答: {str(result)[:100]}...")
            return str(result)
        except Exception as e:
            logger.error(f"RAG invoke 失败: {e}")
            # 降级：直接返回检索到的文档原文
            docs = self._retriever.invoke(question)
            if docs:
                parts = [f"[来源:{d.metadata.get('source_file','知识库')}] {d.page_content}" for d in docs[:3]]
                return "（RAG 生成失败，返回原始检索结果）\n\n" + "\n\n---\n\n".join(parts)
            return f"知识库检索失败: {str(e)[:100]}"

    def stream_query(self, question: str):
        """流式问答，返回生成器（供前端实时展示）。"""
        for chunk in self._chain.stream(question):
            yield chunk


# 全局单例
rag_service = RagService()

"""
services/rag_service.py - RAG 问答服务
基于向量检索 + LLM 回答金融知识库中的问题
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage

from config import settings
from services.vector_store import vector_store_service
from utils.logger import get_logger

logger = get_logger(__name__)

# ── RAG Prompt 模板 ──────────────────────────────────────────
_RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """你是一位专业的金融分析师助手，具备丰富的股票、基金、债券、宏观经济等金融领域知识。
请根据以下【知识库内容】回答用户问题。

回答要求：
1. 优先使用知识库中的信息，引用时请说明"根据知识库..."
2. 如知识库中无相关信息，可结合自身知识回答，但需注明"以下为通用金融知识"
3. 涉及投资建议时，必须在末尾加免责声明："以上内容仅供参考，不构成投资建议，投资有风险，入市须谨慎。"
4. 回答要专业、简洁、有条理，可适当使用列表或小标题

【知识库内容】
{context}
""",
    ),
    MessagesPlaceholder(variable_name="chat_history"),
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
                "chat_history": lambda _: [],   # 无历史时默认空列表
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

        Args:
            question: 用户问题
            chat_history: 历史消息列表，格式 [("human", "..."), ("ai", "..."), ...]

        Returns:
            回答字符串
        """
        # 构建历史消息
        history_messages = []
        for role, content in (chat_history or []):
            if role == "human":
                history_messages.append(HumanMessage(content=content))
            elif role == "ai":
                history_messages.append(AIMessage(content=content))

        # 构建带历史的 Prompt
        prompt_with_history = ChatPromptTemplate.from_messages([
            _RAG_PROMPT.messages[0],   # system
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        chain = (
            {
                "context": self._retriever | _format_docs,
                "question": lambda x: x["question"],
                "chat_history": lambda x: x["chat_history"],
            }
            | prompt_with_history
            | self._llm
            | StrOutputParser()
        )

        result = chain.invoke({
            "question": question,
            "chat_history": history_messages,
        })
        logger.debug(f"RAG 回答: {result[:100]}...")
        return result

    def stream_query(self, question: str):
        """流式问答，返回生成器（供前端实时展示）。"""
        for chunk in self._chain.stream(question):
            yield chunk


# 全局单例
rag_service = RagService()

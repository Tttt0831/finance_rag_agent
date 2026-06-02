"""
agent.py - 金融 Agent 核心组装模块
将 LLM + Tools + Memory + Middleware 组装为可运行的 AgentExecutor
"""

import os
from langchain_openai import ChatOpenAI
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory, SQLChatMessageHistory

from config import settings
from tools import get_all_tools
from middleware.callbacks import FinanceAgentCallback
from utils.logger import get_logger

logger = get_logger(__name__)

# 会话历史存储（内存，支持多用户 session_id 隔离）
_session_store: dict = {}


def _get_session_history(session_id: str) -> ChatMessageHistory:
    """根据 session_id 获取或创建会话历史（内存版）。"""
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


def _get_persistent_session_history(session_id: str) -> SQLChatMessageHistory:
    """根据 session_id 获取或创建会话历史（SQLite 持久化版）。"""
    db_path = os.path.join(settings.log_dir, "chat_history.db")
    return SQLChatMessageHistory(
        session_id=session_id,
        connection_string=f"sqlite:///{db_path}",
    )


def _load_system_prompt() -> str:
    """从文件加载 Agent 系统提示词。"""
    prompt_file = os.path.join(
        os.path.dirname(__file__), "prompts", "agent_system_prompt.txt"
    )
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    # fallback
    return "你是一位专业的金融分析师助手，请使用提供的工具帮助用户解决金融相关问题。"


def create_finance_agent(persistent_memory: bool = False) -> RunnableWithMessageHistory:
    """
    创建金融 Agent（带会话记忆）。

    Args:
        persistent_memory: True=使用 SQLite 持久化记忆，False=使用内存记忆

    Returns:
        带记忆的 RunnableWithMessageHistory Agent
    """
    logger.info("正在创建金融 Agent...")

    # ── 1. 初始化 LLM ────────────────────────────────────────
    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    # ── 2. 加载工具 ──────────────────────────────────────────
    tools = get_all_tools()
    tool_names = [t.name for t in tools]
    logger.info(f"已加载 {len(tools)} 个工具: {tool_names}")

    # ── 3. 构建 ReAct Prompt ─────────────────────────────────
    system_prompt = _load_system_prompt()

    # ReAct 标准 Prompt 格式（必须包含 tools, tool_names, agent_scratchpad）
    react_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            system_prompt + """

## 可用工具

{tools}

## 输出格式（必须严格遵守）

当你需要使用工具时，按以下格式输出（每次只调用一个工具）：
Thought: 我需要思考...（你的推理过程）
Action: 工具名称（必须是 {tool_names} 中的一个）
Action Input: 工具的输入参数

当工具返回结果后，你会看到：
Observation: 工具返回的结果

如果你已经有足够信息回答，输出：
Thought: 我现在知道了...
Final Answer: 你的最终回答（完整、专业）
""",
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])

    # ── 4. 组装 Agent ────────────────────────────────────────
    agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=settings.agent_verbose,
        max_iterations=settings.agent_max_iterations,
        handle_parsing_errors=True,   # 解析错误时自动重试
        callbacks=[FinanceAgentCallback()],
        return_intermediate_steps=False,
    )

    # ── 5. 包装记忆 ──────────────────────────────────────────
    get_history_fn = (
        _get_persistent_session_history
        if persistent_memory
        else _get_session_history
    )

    agent_with_memory = RunnableWithMessageHistory(
        agent_executor,
        get_history_fn,
        input_messages_key="input",
        history_messages_key="chat_history",
    )

    memory_type = "SQLite 持久化" if persistent_memory else "内存"
    logger.info(f"金融 Agent 创建成功（记忆类型: {memory_type}）")
    return agent_with_memory


def get_agent_response(
    agent: RunnableWithMessageHistory,
    user_input: str,
    session_id: str = "default",
) -> str:
    """
    调用 Agent 获取回复。

    Args:
        agent: create_finance_agent() 返回的 Agent 实例
        user_input: 用户输入
        session_id: 会话 ID，用于区分不同用户/对话

    Returns:
        Agent 的回答字符串
    """
    config = {"configurable": {"session_id": session_id}}
    try:
        result = agent.invoke({"input": user_input}, config=config)
        return result.get("output", "抱歉，未能获取到回答，请重试。")
    except Exception as e:
        logger.error(f"Agent 执行出错: {e}")
        return f"处理您的请求时出现错误，请稍后重试。错误信息: {str(e)}"

"""
agent.py - 金融 Agent 核心组装模块
将 LLM + Tools + Memory 组装为可运行的 Agent
使用 LangGraph create_react_agent（原生 tool calling，不再依赖文本解析）
"""

import os
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from config import settings
from tools import get_all_tools
from middleware.callbacks import FinanceAgentCallback
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局记忆（MemorySaver 内置支持多 session_id 隔离）
_memory = MemorySaver()

# 全局回调中间件（日志 / 工具追踪 / token 统计）
# LangGraph 通过 invoke 时的 config["callbacks"] 接入 langchain_core 回调
_callback = FinanceAgentCallback()


def _load_system_prompt() -> str:
    """从文件加载 Agent 系统提示词。"""
    prompt_file = os.path.join(
        os.path.dirname(__file__), "prompts", "agent_system_prompt.txt"
    )
    if os.path.exists(prompt_file):
        with open(prompt_file, "r", encoding="utf-8") as f:
            return f.read().strip()
    return "你是一位专业的金融分析师助手，请使用提供的工具帮助用户解决金融相关问题。"


def create_finance_agent():
    """创建金融 Agent（LangGraph 原生 tool calling + 内存记忆）。"""
    logger.info("正在创建金融 Agent（LangGraph tool-calling 模式）...")

    # ── 0. 配置校验 ──────────────────────────────────────────
    # 提前失败：避免 Key 为空时静默启动，直到首次调用 LLM 才报错
    if not settings.deepseek_api_key:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入真实的 DeepSeek API Key，"
            "获取地址：https://platform.deepseek.com/"
        )

    # ── 1. LLM ──────────────────────────────────────────────
    llm = ChatOpenAI(
        model=settings.model_name,
        api_key=settings.deepseek_api_key,
        base_url=settings.deepseek_base_url,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    # ── 2. 工具 ──────────────────────────────────────────────
    tools = get_all_tools()
    tool_names = [t.name for t in tools]
    logger.info(f"已加载 {len(tools)} 个工具: {tool_names}")

    # ── 3. 系统提示词 ────────────────────────────────────────
    system_prompt = _load_system_prompt()

    # ── 4. 组装 LangGraph Agent ──────────────────────────────
    # LangGraph 的 create_react_agent 使用模型原生 tool calling，
    # 不再需要 ReAct 文本格式解析，彻底解决 markdown 格式冲突
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=_memory,
    )

    logger.info("金融 Agent 创建成功（LangGraph tool-calling + 内存记忆）")
    return agent


def get_agent_response(
    agent,
    user_input: str,
    session_id: str = "default",
) -> str:
    """
    调用 Agent 获取回复。

    Args:
        agent: create_finance_agent() 返回的 LangGraph Agent
        user_input: 用户输入
        session_id: 会话 ID

    Returns:
        Agent 的回答字符串
    """
    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [_callback],
    }
    try:
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        # 提取最后一条 AI 消息
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                return msg.content
        return "抱歉，未能获取到回答，请重试。"
    except Exception as e:
        logger.error(f"Agent 执行出错: {e}")
        return f"处理您的请求时出现错误，请稍后重试。错误信息: {str(e)}"

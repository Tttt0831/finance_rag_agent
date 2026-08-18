"""
agent.py - 金融 Agent 核心组装模块
将 LLM + Tools + Memory 组装为可运行的 Agent
使用 LangGraph create_react_agent（原生 tool calling，不再依赖文本解析）

生产化升级（对照《Agent 开发知识库》）：
- 记忆落盘：MemorySaver → SqliteSaver（断点续跑 / 时间旅行）；
- 熔断：recursion_limit 步数上限 + 回调内 token 预算熔断；
- 上下文 compaction：token 超水位时压缩旧轮次为摘要；
- 流式：astream_events 按 token/tool 事件产出；
- 可观测：会话 trace 在结束时落盘 JSONL。
"""

import os

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, RemoveMessage
from langgraph.prebuilt import create_react_agent

from config import settings
from tools import get_all_tools
from middleware.callbacks import FinanceAgentCallback, BudgetExceeded
from middleware.trace import set_trace_thread_id, trace_store
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局 LLM（供 compaction 摘要复用），create_finance_agent 时赋值
_llm = None

# 全局回调中间件（日志 / 工具追踪 / token 统计 / 熔断）
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


def _create_checkpointer():
    """创建会话状态存储：优先 SqliteSaver（落盘），不可用时回退 MemorySaver。"""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
        return SqliteSaver.from_conn_string(settings.checkpoint_db_path)
    except Exception as e:
        logger.warning(f"SqliteSaver 不可用，回退 MemorySaver（不落盘）: {e}")
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


def create_finance_agent():
    """创建金融 Agent（LangGraph 原生 tool calling + 落盘记忆 + 熔断）。"""
    global _llm
    logger.info("正在创建金融 Agent（LangGraph tool-calling 模式）...")

    # ── 0. 配置校验 ──────────────────────────────────────────
    # 提前失败：避免 Key 为空时静默启动，直到首次调用 LLM 才报错
    if not settings.deepseek_api_key:
        raise RuntimeError(
            "未配置 DEEPSEEK_API_KEY。请复制 .env.example 为 .env 并填入真实的 DeepSeek API Key，"
            "获取地址：https://platform.deepseek.com/"
        )

    # ── 1. LLM ──────────────────────────────────────────────
    _llm = ChatOpenAI(
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
    # recursion_limit = 步数熔断（KB 模块 01 终止条件 3）
    agent = create_react_agent(
        model=_llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=_create_checkpointer(),
        recursion_limit=settings.max_steps,
    )

    logger.info(
        f"金融 Agent 创建成功（tool-calling + 落盘记忆 + 步数上限 {settings.max_steps}）"
    )
    return agent


# ────────────────────────────────────────────────────────────────
# 上下文 compaction（KB 模块 05）
# ────────────────────────────────────────────────────────────────

def _estimate_tokens(messages) -> int:
    """粗略 token 估计：中文约 1 字符 ≈ 1 token，按字符数近似。"""
    return sum(len(str(m.content)) for m in messages)


def _summarize_history(messages) -> str:
    """把旧轮次压成一段摘要（保留目标/已完成/未决/约束四件套）。"""
    global _llm
    if _llm is None:
        return ""
    text = "\n".join(f"{getattr(m, 'type', '?')}: {m.content}" for m in messages)
    prompt = (
        "请把以下对话历史压缩成一段中文摘要，务必保留：任务目标、已完成步骤、"
        "未决问题、关键约束。只输出摘要正文。\n\n" + text[:6000]
    )
    try:
        resp = _llm.invoke(prompt)
        return str(resp.content).strip()
    except Exception as e:
        logger.warning(f"[compaction] 摘要生成失败: {e}")
        return ""


def _maybe_compact(agent, config) -> None:
    """token 超水位时，压缩旧轮次为摘要并回写状态。"""
    try:
        state = agent.get_state(config)
        messages = state.values.get("messages", []) if state.values else []
    except Exception as e:
        logger.debug(f"[compaction] 读取状态失败，跳过: {e}")
        return

    if _estimate_tokens(messages) < settings.compaction_threshold_tokens:
        return

    keep = 4  # 保留最后两轮
    old, recent = messages[:-keep], messages[-keep:]
    if not old:
        return

    summary = _summarize_history(old)
    if not summary:
        return

    # messages 通道的 reducer 是 add_messages（追加），故用 RemoveMessage 显式删除旧消息，
    # 再追加摘要 + 近期消息，实现真正的「压缩替换」而非追加。
    removals = [RemoveMessage(id=m.id) for m in old if getattr(m, "id", None)]
    new_messages = [SystemMessage(content="[历史摘要] " + summary)] + list(recent)
    try:
        agent.update_state(config, {"messages": removals + new_messages})
        logger.info(f"[compaction] 压缩 {len(old)} 条历史 → 摘要（{len(summary)} 字）")
    except Exception as e:
        logger.warning(f"[compaction] 回写状态失败: {e}")


# ────────────────────────────────────────────────────────────────
# 调用入口
# ────────────────────────────────────────────────────────────────

def get_agent_response(
    agent,
    user_input: str,
    session_id: str = "default",
) -> str:
    """调用 Agent 获取回复（含 compaction + token 熔断 + trace 落盘）。"""
    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [_callback],
    }
    set_trace_thread_id(session_id)
    try:
        _maybe_compact(agent, config)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        messages = result.get("messages", [])
        for msg in reversed(messages):
            if hasattr(msg, "content") and msg.type == "ai" and msg.content:
                return msg.content
        return "抱歉，未能获取到回答，请重试。"
    except BudgetExceeded as e:
        logger.warning(f"[熔断] {e}")
        return "（会话 token 预算已用尽，本次对话到此为止，请新建会话继续。）"
    except Exception as e:
        logger.error(f"Agent 执行出错: {e}")
        return f"处理您的请求时出现错误，请稍后重试。错误信息: {str(e)}"
    finally:
        trace_store.dump(session_id)
        trace_store.clear(session_id)


async def stream_agent(agent, user_input: str, session_id: str = "default"):
    """
    流式调用：按 token / tool 事件产出（KB 模块 07 SSE 骨架的异步版）。
    供 Gradio 等前端逐字渲染；非流式 get_agent_response 仍为默认路径。
    """
    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [_callback],
    }
    set_trace_thread_id(session_id)
    _maybe_compact(agent, config)
    try:
        async for event in agent.astream_events(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
            version="v2",
        ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                chunk = event["data"].get("chunk")
                content = getattr(chunk, "content", None)
                if content:
                    yield {"event": "token", "data": content}
            elif kind == "on_tool_start":
                yield {"event": "tool", "data": event.get("name")}
    finally:
        trace_store.dump(session_id)
        trace_store.clear(session_id)

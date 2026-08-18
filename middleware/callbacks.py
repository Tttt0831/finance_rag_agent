"""
middleware/callbacks.py - Agent 中间件（回调处理器）
负责：日志记录 / 工具调用追踪 / 执行时间统计 / token 预算熔断 / trace 落盘

接入方式：在 agent.invoke 时通过 config={"callbacks": [FinanceAgentCallback()]}
传入（见 agent.py:get_agent_response）。LangGraph 复用 langchain_core 的回调机制，
on_llm_* / on_tool_* 会正常触发；on_agent_action / on_agent_finish 是 langchain_classic
AgentExecutor 的钩子，在 LangGraph 下不会被调用，仅作兼容保留。

token 预算按 thread_id 隔离累计（回调是全局单例，跨会话复用），超过
settings.max_total_tokens 时抛 BudgetExceeded，由 agent.get_agent_response 捕获熔断。
"""

import time
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from config import settings
from middleware.trace import get_trace_thread_id, trace_store
from utils.logger import get_logger

logger = get_logger(__name__)


class BudgetExceeded(Exception):
    """会话累计 token 超过预算的熔断信号。"""


class FinanceAgentCallback(BaseCallbackHandler):
    """金融 Agent 全链路日志中间件 + token 预算熔断。"""

    def __init__(self):
        super().__init__()
        self._tool_start_time: Dict[str, float] = {}
        self._llm_start_time: float = 0.0
        # 按 thread_id 隔离的累计 token（会话级预算）
        self._thread_tokens: Dict[str, int] = {}

    @staticmethod
    def _thread() -> str:
        return get_trace_thread_id() or "unknown"

    # ── LLM 调用 ────────────────────────────────────────────

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs):
        self._llm_start_time = time.time()
        serialized = serialized or {}
        model = serialized.get("kwargs", {}).get("model_name", "unknown")
        logger.debug(f"[LLM 开始] 模型: {model}, Prompt 数量: {len(prompts)}")

    def on_llm_end(self, response: LLMResult, **kwargs):
        elapsed = time.time() - self._llm_start_time
        token_usage = (
            response.llm_output.get("token_usage", {})
            if response.llm_output else {}
        )
        total_tokens = token_usage.get("total_tokens", 0)

        thread = self._thread()
        self._thread_tokens[thread] = self._thread_tokens.get(thread, 0) + total_tokens
        cum = self._thread_tokens[thread]
        trace_store.record_llm(thread, total_tokens, round(elapsed, 3))
        logger.info(
            f"[LLM 完成] 耗时: {elapsed:.2f}s  "
            f"本次 Token: {total_tokens}  会话累计 Token: {cum}"
        )

        # token 预算熔断（KB 模块 01 终止条件 2）
        if cum > settings.max_total_tokens:
            logger.warning(f"[熔断] 会话累计 token {cum} 超过预算 {settings.max_total_tokens}")
            raise BudgetExceeded(
                f"会话累计 token {cum} 超过预算 {settings.max_total_tokens}"
            )

    def on_llm_error(self, error: Exception, **kwargs):
        logger.error(f"[LLM 错误] {type(error).__name__}: {error}")

    # ── 工具调用 ────────────────────────────────────────────

    def on_tool_start(self, serialized: Dict, input_str: str, **kwargs):
        serialized = serialized or {}
        tool_name = serialized.get("name", "unknown_tool")
        self._tool_start_time[tool_name] = time.time()
        trace_store.record_tool(self._thread(), tool_name, 0.0)
        logger.info(f"[工具调用] ▶ {tool_name}")
        logger.debug(f"[工具输入] {input_str[:200]}")

    def on_tool_end(self, output: str, **kwargs):
        output_preview = str(output)[:150] + ("..." if len(str(output)) > 150 else "")
        logger.info(f"[工具完成] ✓")
        logger.debug(f"[工具输出] {output_preview}")

    def on_tool_error(self, error: Exception, **kwargs):
        logger.error(f"[工具错误] {type(error).__name__}: {error}")

    # ── Agent 推理（langchain_classic 兼容保留）─────────────

    def on_agent_action(self, action, **kwargs):
        logger.info(
            f"[Agent 决策] 工具: {action.tool}  "
            f"思考: {str(action.log)[:100].strip()}"
        )

    def on_agent_finish(self, finish, **kwargs):
        output_preview = str(finish.return_values.get("output", ""))[:100]
        logger.info(f"[Agent 完成] 输出预览: {output_preview}")

    # ── 链 ──────────────────────────────────────────────────

    def on_chain_start(self, serialized: Dict, inputs: Dict, **kwargs):
        serialized = serialized or {}
        chain_name = serialized.get("id", ["unknown"])[-1]
        logger.debug(f"[链开始] {chain_name}")

    def on_chain_error(self, error: Exception, **kwargs):
        logger.error(f"[链错误] {type(error).__name__}: {error}")

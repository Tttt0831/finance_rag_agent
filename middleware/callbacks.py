"""
middleware/callbacks.py - Agent 中间件（回调处理器）
负责：日志记录 / 工具调用追踪 / 执行时间统计

接入方式：在 agent.invoke 时通过 config={"callbacks": [FinanceAgentCallback()]}
传入（见 agent.py:get_agent_response）。LangGraph 复用 langchain_core 的回调机制，
on_llm_* / on_tool_* 会正常触发；on_agent_action / on_agent_finish 是 langchain_classic
AgentExecutor 的钩子，在 LangGraph 下不会被调用，仅作兼容保留。
"""

import time
from typing import Any, Dict, List

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

from utils.logger import get_logger

logger = get_logger(__name__)


class FinanceAgentCallback(BaseCallbackHandler):
    """
    金融 Agent 全链路日志中间件。
    记录 LLM 调用、工具调用、执行耗时等关键信息。
    """

    def __init__(self):
        super().__init__()
        self._tool_start_time: Dict[str, float] = {}
        self._llm_start_time: float = 0.0
        self._session_token_count: int = 0

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
        self._session_token_count += total_tokens
        logger.info(
            f"[LLM 完成] 耗时: {elapsed:.2f}s  "
            f"本次 Token: {total_tokens}  累计 Token: {self._session_token_count}"
        )

    def on_llm_error(self, error: Exception, **kwargs):
        logger.error(f"[LLM 错误] {type(error).__name__}: {error}")

    # ── 工具调用 ────────────────────────────────────────────

    def on_tool_start(self, serialized: Dict, input_str: str, **kwargs):
        serialized = serialized or {}
        tool_name = serialized.get("name", "unknown_tool")
        self._tool_start_time[tool_name] = time.time()
        logger.info(f"[工具调用] ▶ {tool_name}")
        logger.debug(f"[工具输入] {input_str[:200]}")

    def on_tool_end(self, output: str, **kwargs):
        output_preview = str(output)[:150] + ("..." if len(str(output)) > 150 else "")
        logger.info(f"[工具完成] ✓")
        logger.debug(f"[工具输出] {output_preview}")

    def on_tool_error(self, error: Exception, **kwargs):
        logger.error(f"[工具错误] {type(error).__name__}: {error}")

    # ── Agent 推理 ──────────────────────────────────────────

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

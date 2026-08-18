"""
middleware/trace.py - 会话级结构化 trace（可观测性）
按 thread_id 记录 llm_call / tool_call / retrieval 三类 span。
检索 span 额外记录 doc_id + source_file + score —— 排查「没召回到 vs 召回了没用」的关键。

设计：
- TraceStore 单例，内存累计，会话结束时由 agent 落盘 JSONL。
- 通过 ContextVar 把 thread_id 穿透到工具内部的检索调用（检索发生在 tool 函数内部，
  不在 langchain callback 覆盖范围，故不能依赖 on_tool_* 拿到 doc 细节）。
"""

import json
import os
import threading
from contextvars import ContextVar
from typing import Any, Dict, List, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

# 穿透 thread_id：agent 在 invoke 前 set，检索器在记录时 get
_current_thread_id: ContextVar[Optional[str]] = ContextVar(
    "trace_thread_id", default=None
)


def set_trace_thread_id(thread_id: str) -> None:
    """设置当前调用链所属的会话 ID（agent 在 invoke 前调用）。"""
    _current_thread_id.set(thread_id)


def get_trace_thread_id() -> Optional[str]:
    return _current_thread_id.get()


class TraceStore:
    """会话 trace 内存存储（单例）。"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._traces: Dict[str, Dict[str, Any]] = {}
        return cls._instance

    # ── 记录 ────────────────────────────────────────────────

    def record_llm(self, thread_id: str, tokens: int, latency: float,
                   model: str = "unknown") -> None:
        self._append(thread_id, {"type": "llm_call", "tokens": tokens,
                                 "latency": latency, "model": model})

    def record_tool(self, thread_id: str, name: str, latency: float,
                    error: Optional[str] = None) -> None:
        self._append(thread_id, {"type": "tool_call", "name": name,
                                 "latency": latency, "error": error})

    def record_retrieval(self, thread_id: str, query: str,
                         docs: List[Dict[str, Any]], method: str = "hybrid") -> None:
        """记录一次检索：query + 命中块（含 doc_id / source_file / score）。"""
        hits = []
        for d in docs:
            meta = d.get("metadata", {}) if isinstance(d, dict) else {}
            hits.append({
                "doc_id": meta.get("id") or meta.get("parent_id") or meta.get("source_file"),
                "source_file": meta.get("source_file"),
                "score": d.get("score"),
            })
        self._append(thread_id, {"type": "retrieval", "query": query,
                                 "method": method, "hits": hits})

    # ── 读取 / 落盘 / 清理 ─────────────────────────────────

    def _append(self, thread_id: str, span: Dict[str, Any]) -> None:
        with self._lock:
            tr = self._traces.setdefault(thread_id, {"spans": []})
            tr["spans"].append(span)

    def get_trace(self, thread_id: str) -> Dict[str, Any]:
        return self._traces.get(thread_id, {"spans": []})

    def dump(self, thread_id: str) -> Optional[str]:
        """将会话 trace 落盘 JSONL，返回文件路径（无 span 时返回 None）。"""
        tr = self._traces.get(thread_id)
        if not tr or not tr.get("spans"):
            return None
        path = os.path.join(settings.trace_dir, f"{thread_id[:32]}.jsonl")
        try:
            with open(path, "a", encoding="utf-8") as f:
                for span in tr["spans"]:
                    f.write(json.dumps(span, ensure_ascii=False) + "\n")
            return path
        except Exception as e:
            logger.error(f"trace 落盘失败: {e}")
            return None

    def clear(self, thread_id: str) -> None:
        with self._lock:
            self._traces.pop(thread_id, None)


# 全局单例
trace_store = TraceStore()

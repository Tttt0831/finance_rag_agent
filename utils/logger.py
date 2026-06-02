"""
utils/logger.py - 统一日志工具
输出到控制台 + 文件，支持日志级别配置
"""

import logging
import os
from datetime import datetime
from config import settings


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的 Logger。
    所有模块通过此函数获取 logger，保证统一格式。

    Args:
        name: 模块名称，通常传入 __name__

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    # ── 格式 ────────────────────────────────────────────────
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── 控制台 Handler ──────────────────────────────────────
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── 文件 Handler ────────────────────────────────────────
    log_file = os.path.join(
        settings.log_dir,
        f"finance_agent_{datetime.now().strftime('%Y%m%d')}.log",
    )
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

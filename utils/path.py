"""
utils/path.py - 路径工具
"""

import os
import hashlib


def ensure_dir(path: str) -> str:
    """确保目录存在，不存在则创建。返回路径本身。"""
    os.makedirs(path, exist_ok=True)
    return path


def get_file_md5(file_bytes: bytes) -> str:
    """计算文件字节流的 MD5，用于文件去重。"""
    return hashlib.md5(file_bytes).hexdigest()


def get_project_root() -> str:
    """获取项目根目录的绝对路径。"""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def resolve_path(*parts: str) -> str:
    """以项目根目录为基准，拼接相对路径。"""
    return os.path.join(get_project_root(), *parts)

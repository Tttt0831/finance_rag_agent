"""
utils/path.py - 路径工具
"""

import hashlib


def get_file_md5(file_bytes: bytes) -> str:
    """计算文件字节流的 MD5，用于文件去重。"""
    return hashlib.md5(file_bytes).hexdigest()

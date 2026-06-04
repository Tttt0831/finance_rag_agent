"""
config.py - 全局配置管理
使用 pydantic-settings 从环境变量读取配置，支持 .env 文件
"""

import os
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # ── 大模型配置 ──────────────────────────────────────────
    model_name: str = Field(default="deepseek-chat", description="LLM 模型名称")
    temperature: float = Field(default=0.3, description="温度参数，金融场景建议偏低以保证准确性")
    max_tokens: int = Field(default=2048, description="最大输出 Token 数")

    # DeepSeek API（OpenAI 兼容接口）
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = "https://api.deepseek.com/v1"

    # 嵌入模型（本地 HuggingFace 模型，无需 API Key）
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # ── 向量库配置 ──────────────────────────────────────────
    chroma_persist_dir: str = Field(default="./data/chroma_db", env="CHROMA_PERSIST_DIR")
    retriever_k: int = Field(default=4, description="RAG 检索返回 Top-K 文档数")
    chunk_size: int = Field(default=600, description="文本分块大小")
    chunk_overlap: int = Field(default=80, description="分块重叠字符数")

    # ── 日志配置 ──────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_dir: str = "./logs"

    # ── 数据路径 ──────────────────────────────────────────
    knowledge_dir: str = "./data/knowledge"   # 金融知识文档目录
    upload_dir: str = "./data/uploads"         # 用户上传文档目录

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # 忽略 .env 中未匹配的变量


# 全局单例
settings = Settings()

# 确保必要目录存在
for _dir in [
    settings.chroma_persist_dir,
    settings.log_dir,
    settings.knowledge_dir,
    settings.upload_dir,
]:
    os.makedirs(_dir, exist_ok=True)

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

    # ── 混合检索（BM25 + RRF）配置 ─────────────────────────
    enable_hybrid: bool = Field(default=True, description="是否启用 BM25 + 稠密检索的混合召回")
    hybrid_k: int = Field(default=20, description="混合召回时的候选池大小（两路各取 top-N）")
    rrf_k: int = Field(default=60, description="RRF 融合常数")
    enable_query_rewrite: bool = Field(default=False, description="是否启用查询改写（多意图分解），默认关以省一次 LLM 调用")

    # ── Rerank 精排（P3，默认关，避免强制下载大模型）───────
    enable_rerank: bool = Field(default=False, description="是否启用 cross-encoder 精排")
    rerank_model: str = Field(default="BAAI/bge-reranker-base", description="精排模型")
    rerank_top_k: int = Field(default=8, description="精排候选数（RRF 后送入 rerank 的条数）")

    # ── 父子块切分（small-to-big，P3，默认关，开启需重建索引）──
    enable_parent_child: bool = Field(default=False, description="是否启用父子块切分")
    child_chunk_size: int = Field(default=256, description="子块大小（用于检索）")
    parent_chunk_size: int = Field(default=1200, description="父块大小（用于喂给模型）")

    # ── Agent 运行控制（熔断 / 压缩 / 落盘）────────────────
    max_steps: int = Field(default=8, description="Agent 循环最大步数（recursion_limit）")
    max_total_tokens: int = Field(default=40000, description="单会话累计 token 预算，超限熔断")
    compaction_threshold_tokens: int = Field(default=8000, description="触发上下文压缩的 token 水位")
    checkpoint_db_path: str = Field(default="./data/checkpoints.sqlite", description="会话状态落盘路径")

    # ── 评测配置 ──────────────────────────────────────────
    golden_set_path: str = Field(default="./eval/golden_set.json", description="金标集路径")

    # ── 日志配置 ──────────────────────────────────────────
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    log_dir: str = "./logs"
    trace_dir: str = Field(default="./logs/traces", description="会话 trace 落盘目录")

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
    settings.trace_dir,
    settings.knowledge_dir,
    settings.upload_dir,
]:
    os.makedirs(_dir, exist_ok=True)

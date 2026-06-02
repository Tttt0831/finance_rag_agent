# 💹 财智助手 · 金融 Agent 项目

基于 **LangChain + RAG + ReAct Agent** 构建的 AI 金融分析师。

- **LLM**: DeepSeek（OpenAI 兼容接口）
- **Embedding**: HuggingFace 本地模型 `BAAI/bge-small-zh-v1.5`（免费，无需 API Key）
- **前端**: Gradio 6.x 聊天界面
- **API 服务**: FastAPI（可选）

> 🧩 **入门学习**：想快速理解整个架构？查看 [finance_agent_mini.py](finance_agent_mini.py) —— 同一架构，单文件 ~280 行，7 层注解，3 分钟读完全流程。

---

## 📁 项目结构

```
finance_agent/
├── .env / .env.example   # 环境变量（API Key 等）
├── requirements.txt      # 依赖清单
├── config.py             # 全局配置管理（pydantic-settings）
├── finance_agent_mini.py # 🔥 超级缩略版（单文件蓝图，入门学习用）
│
├── utils/                # 工具函数
│   ├── logger.py         # 统一日志（输出到控制台 + 文件）
│   └── path.py           # 路径工具（MD5去重、路径拼接）
│
├── services/             # 核心服务
│   ├── vector_store.py   # 向量存储服务（文档入库、相似度检索）
│   └── rag_service.py    # RAG 问答服务（检索 + LLM 生成）
│
├── tools/
│   └── finance_tools.py  # Agent 工具集（8个金融工具）
│
├── middleware/
│   └── callbacks.py      # 日志中间件（LLM/工具调用全链路追踪）
│
├── prompts/
│   └── agent_system_prompt.txt  # Agent 系统提示词
│
├── agent.py              # Agent 核心组装（LLM + Tools + Memory）
├── app.py                # Gradio 前端界面（主入口）
├── server.py             # FastAPI REST API（可选，独立部署）
│
└── data/
    ├── finance_knowledge.txt  # 内置金融知识文档（自动加载）
    ├── uploads/               # 用户上传文档存储目录
    ├── knowledge/             # 额外知识文档目录
    ├── chroma_db/             # 向量数据库持久化目录（自动创建）
    └── .indexed_md5.txt       # 已入库文件的 MD5 记录（去重用）
```

---

## 🚀 快速开始

### 1. 安装依赖

```bash
# 建议使用 Python 3.10+
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
# 编辑 .env，填入你的 DeepSeek API Key
```

`.env` 内容示例：
```
DEEPSEEK_API_KEY=sk-your-api-key-here
```

> 在 [DeepSeek 开放平台](https://platform.deepseek.com/) 获取 API Key。

### 3. 启动 Gradio 前端（主入口）

```bash
python app.py
```

浏览器访问 `http://localhost:7860` 即可使用。

### 4. （可选）启动 FastAPI 接口服务

```bash
python server.py
# 接口文档：http://localhost:8000/docs
```

### 5. 🧩 入门学习：运行缩略版

```bash
python finance_agent_mini.py
```

一个文件，~280 行，完整演示项目 7 层架构。详细注解请直接阅读 [finance_agent_mini.py](finance_agent_mini.py)。

---

## 🤖 Agent 工具列表

| 工具名 | 功能 | 数据来源 |
|--------|------|----------|
| `search_knowledge_base` | 检索金融知识库（RAG）| 本地 Chroma 向量库 |
| `get_stock_realtime` | A股实时行情（价格/涨跌/市值）| AKShare |
| `get_stock_history` | 股票历史K线数据 | AKShare |
| `get_stock_financial` | 财务指标（PE/PB/换手率）| AKShare |
| `get_market_index` | 主要市场指数实时行情 | AKShare |
| `get_fund_info` | 公募基金净值与表现 | AKShare |
| `calculate_financial` | 金融计算器（复利/收益率/仓位）| 本地计算 |
| `get_current_datetime` | 当前时间与交易日判断 | 系统时间 |

---

## 💬 使用示例

**查询股票行情：**
```
用户: 帮我查一下贵州茅台的今天股价
Agent: [调用 get_stock_realtime("600519")]
       贵州茅台（600519）最新价：1668.00元，涨跌幅：+1.23%...
```

**金融知识问答（RAG）：**
```
用户: 什么是市盈率，多少算合理？
Agent: [调用 search_knowledge_base("市盈率")]
       根据知识库：市盈率 = 股价 / 每股收益，反映...
```

**金融计算：**
```
用户: 我用10万本金，年化8%，投资10年能有多少？
Agent: [调用 calculate_financial(calc_type="compound", principal=100000, rate=8, years=10)]
       复利计算结果：最终金额：215,892.50 元，总收益：115,892.50 元...
```

**上传私有文档后问答：**
```
# 上传一份公司年报 PDF
# Agent 会基于年报内容回答问题
用户: 这份年报里公司的净利润增速是多少？
Agent: [调用 search_knowledge_base("净利润增速")]
       根据知识库（来源：annual_report.pdf）...
```

---

## ⚙️ 配置说明（config.py）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | `deepseek-chat` | LLM 模型，DeepSeek OpenAI 兼容接口 |
| `temperature` | `0.3` | 金融场景建议偏低（更准确），创意场景可调高到 0.7 |
| `embedding_model` | `BAAI/bge-small-zh-v1.5` | 本地中文嵌入模型（HuggingFace，免费） |
| `retriever_k` | `4` | RAG 检索返回的文档块数量 |
| `chunk_size` | `600` | 文本分块大小（字符数） |
| `agent_max_iterations` | `8` | Agent 最大工具调用次数 |
| `deepseek_base_url` | `https://api.deepseek.com/v1` | DeepSeek API 地址（兼容 OpenAI 格式） |

---

## 📌 注意事项

1. **数据时效性**：股票/基金数据通过 AKShare 实时获取，非交易时间可能无最新数据；东方财富 API 在不稳定网络环境（如代理/VPN）下可能连接失败
2. **投资免责**：本工具所有输出仅供参考，不构成投资建议
3. **API 配额**：DeepSeek API 有调用限制，请合理使用
4. **向量库持久化**：向量数据存储在 `./data/chroma_db/`，删除该目录会清空知识库；更换 Embedding 模型后需清空重建
5. **首次启动**：HuggingFace 嵌入模型会在首次运行时自动下载（约 100MB），请确保网络畅通

---

## 🏗️ 技术架构

```
用户输入
    │
    ▼
Gradio 前端 (app.py)
    │
    ▼
RunnableWithMessageHistory（带记忆）
    │
    ▼
AgentExecutor（ReAct 框架）
    │
    ├── [工具选择] ──► 8 个金融工具
    │                    ├── RAG 知识库检索 (Chroma + HuggingFace Embedding)
    │                    ├── AKShare 金融数据 API
    │                    └── 本地计算函数
    │
    ├── [LLM 推理] ──► DeepSeek (via OpenAI 兼容接口)
    │
    └── [日志中间件] ── FinanceAgentCallback (全链路追踪)
```

---

## 🧩 缩略版对照

| 完整项目 (10 文件) | 缩略版 (1 文件) | 层级 |
|---------------------|-----------------|------|
| `.env` / `config.py` | `os.getenv()` + 常量 | 第 1 层 — 配置 |
| `services/vector_store.py` + `services/rag_service.py` | `vectorstore` + `_rag_query()` | 第 2+4 层 — 模型 + RAG |
| `tools/finance_tools.py` (8 工具) | `@tool` × 3 | 第 3 层 — 工具集 |
| `agent.py` + `middleware/callbacks.py` | `create_agent()` + `_get_history()` | 第 5 层 — Agent |
| `app.py` (Gradio Blocks) | `gr.ChatInterface` | 第 6 层 — UI |
| `server.py` (FastAPI) | （省略） | — |

查看 [finance_agent_mini.py](finance_agent_mini.py) 获取完整逐行注解。

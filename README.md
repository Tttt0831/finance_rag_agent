# 💹 财智助手 — AI 金融分析师 Agent

基于 **LangChain + LangGraph + DeepSeek + RAG** 的智能金融分析助手。

```
你: "帮我分析一下茅台的估值"       你: "10万投8%的年化，10年后多少？"
Agent: [调取实时股价] →             Agent: [调用复利计算器] →
       [检索知识库: PE/估值] →             "最终 215,892 元，收益 115.89%"
       "茅台 PE 19.2，处于合理区间..."
```

---

## 🧩 三层学习路径

| 文件 | 行数 | 用途 | 适合 |
|------|------|------|------|
| **[finance_agent_skeleton.py](finance_agent_skeleton.py)** | ~280 | 填空学习版，6 个 STEP 渐进式 TODO | 🟢 入门 |
| **[finance_agent_mini.py](finance_agent_mini.py)** | ~260 | 完整参考实现，逐段注解 | 🟡 理解 |
| **完整项目** (app.py + agent.py + tools/...) | ~800 | 生产级架构，8 工具 + FastAPI | 🔴 深入 |

> **建议顺序**: skeleton（自己动手）→ mini（对照答案）→ 完整项目（看工程化）

---

## 📁 项目结构

```
finance-agent/
├── 📖 学习文件
│   ├── finance_agent_skeleton.py  # 填空学习版 (6 STEP 渐进式)
│   └── finance_agent_mini.py      # 完整参考实现 (对应 skeleton 答案)
│
├── 🚀 生产代码
│   ├── app.py                     # Gradio 前端 (主入口)
│   ├── agent.py                   # Agent 组装 (LangGraph)
│   ├── server.py                  # FastAPI REST API
│   ├── config.py                  # 全局配置 (pydantic-settings)
│   │
│   ├── tools/
│   │   └── finance_tools.py       # 8 个金融工具 (腾讯财经 API)
│   │
│   ├── services/
│   │   ├── vector_store.py        # ChromaDB 向量存储
│   │   └── rag_service.py         # RAG 检索增强生成
│   │
│   ├── middleware/
│   │   └── callbacks.py           # 日志回调 (LangGraph 待适配)
│   │
│   ├── utils/
│   │   ├── logger.py              # 统一日志
│   │   └── path.py                # 路径工具
│   │
│   └── prompts/
│       └── agent_system_prompt.txt # Agent 系统提示词
│
├── data/
│   ├── finance_knowledge.txt      # 内置金融知识
│   └── uploads/                   # 用户上传文档
│
├── .env.example                   # 环境变量模板
├── requirements.txt               # Python 依赖
└── README.md                      # 本文件
```

---

## 🚀 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

```bash
cp .env.example .env
```

编辑 `.env`:
```
DEEPSEEK_API_KEY=sk-your-key-here
```

> 在 [DeepSeek 开放平台](https://platform.deepseek.com/) 获取 API Key

### 3. 启动

```bash
# 学习版（推荐先跑这个）
python finance_agent_mini.py

# 完整版
python app.py

# 可选: FastAPI 服务
python server.py    # http://localhost:8000/docs
```

浏览器访问 `http://localhost:7860`

---

## 🤖 8 个 Agent 工具

| 工具 | 功能 | 数据来源 |
|------|------|----------|
| `search_knowledge_base` | 检索金融知识库 | 本地 ChromaDB + LLM 总结 |
| `get_stock_realtime` | A股实时行情 (价格/涨跌/PE/市值) | 腾讯财经 `qt.gtimg.cn` |
| `get_stock_history` | 历史 K 线 (日/周/月线) | 腾讯财经 `web.ifzq.gtimg.cn` |
| `get_stock_financial` | 财务指标 (PE/市值/换手率) | 腾讯财经 |
| `get_market_index` | 上证/深证/创业板/科创50/沪深300 | 腾讯财经 |
| `get_fund_info` | 基金净值 | 天天基金 `1234567.com.cn` |
| `calculate_financial` | 复利/收益率/PE/PB/仓位计算 | 本地计算 |
| `get_current_datetime` | 当前时间 + 交易日判断 | 系统时间 |

---

## 🏗️ 架构深度解析

### Agent 的生命周期

```
 ┌──────────────────────────────────────────────────────┐
 │                    用户输入                           │
 │              "茅台 PE 多少？合理吗？"                    │
 └──────────────────┬───────────────────────────────────┘
                    ▼
 ┌──────────────────────────────────────────────────────┐
 │  LangGraph Agent (create_react_agent)                 │
 │                                                      │
 │  ┌─────────┐   ┌──────────┐   ┌──────────────────┐   │
 │  │  LLM    │──▶│ 需要工具？│──▶│ 调用 get_stock_  │   │
 │  │ DeepSeek│   │          │   │ realtime("600519")│   │
 │  └─────────┘   └──────────┘   └───────┬──────────┘   │
 │       ▲                    │           │              │
 │       │              直接回答          │ 工具结果      │
 │       │                    │           ▼              │
 │       │              ┌──────────┐  ┌──────────────┐  │
 │       └──────────────│ 生成回复 │◀─│ 调用 search_ │  │
 │                      │          │  │ knowledge()  │  │
 │                      └──────────┘  └──────────────┘  │
 └──────────────────────────────────────────────────────┘
                    ▼
              "茅台 PE 19.2，处于合理估值区间..."
```

### 核心组件职责

| 组件 | 一句话 | 技术 |
|------|--------|------|
| **LLM** | 大脑 - 理解问题、决策、生成回答 | DeepSeek (OpenAI 兼容) |
| **Tools** | 手和眼 - 获取外部数据、执行计算 | `@tool` + 腾讯 API |
| **RAG** | 记忆库 - 检索私有文档、专业概念 | ChromaDB + HuggingFace Embedding |
| **Agent** | 调度中心 - 协调 LLM + Tools 的循环推理 | LangGraph `create_react_agent` |
| **Memory** | 对话历史 - 让 Agent 记住之前的上下文 | LangGraph `MemorySaver` |
| **UI** | 界面 - 浏览器中的聊天窗口 | Gradio `ChatInterface` |

### 为什么用 LangGraph 而不是 langchain_classic？

| | langchain_classic | LangGraph |
|------|------|------|
| 工具调用 | 文本格式解析 (解析 `Action:` `Action Input:`) | 原生 function calling (JSON) |
| 可靠性 | 模型格式稍偏就解析失败 | 不存在解析问题 |
| 记忆 | `RunnableWithMessageHistory` (已弃用) | `MemorySaver` (内置持久化) |
| 控制流 | 线性的 ReAct 循环 | 可自定义的图结构 |

---

## 📊 数据流全景

```
                    ┌──────────────┐
                    │   Gradio UI  │  ← 用户浏览器
                    └──────┬───────┘
                           │ chat()
                           ▼
              ┌────────────────────────┐
              │    LangGraph Agent      │
              │  (create_react_agent)   │
              │                        │
              │  ┌──────────────────┐  │
              │  │   DeepSeek LLM   │  │  ← 推理引擎
              │  └──────────────────┘  │
              │         │              │
              │    tool calling?       │
              │    ┌────┴────┐         │
              │    ▼         ▼         │
              │ ┌──────┐ ┌──────┐      │
              │ │腾讯API│ │Chroma│      │  ← 数据源
              │ │股票/基│ │ RAG  │      │
              │ │金/指数│ │知识库│      │
              │ └──────┘ └──────┘      │
              │    │         │         │
              │    └────┬────┘         │
              │         ▼              │
              │  ┌──────────────┐      │
              │  │ 组装最终回复  │      │  ← 综合结果
              │  └──────────────┘      │
              └────────────────────────┘
                           │
                           ▼
                     用户看到回复
```

---

## ⚙️ 配置说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `DEEPSEEK_API_KEY` | (必填) | DeepSeek API Key |
| `model_name` | `deepseek-chat` | LLM 模型 |
| `temperature` | `0.3` | 温度参数 (金融场景偏低) |
| `embedding_model` | `BAAI/bge-small-zh-v1.5` | HuggingFace 嵌入模型 |
| `retriever_k` | `4` | RAG 检索文档数 |
| `chunk_size` | `600` | 文本分块大小 |
| `CHROMA_PERSIST_DIR` | `./data/chroma_db` | 向量库路径 |

---

## 🔧 常见问题

**Q: 股票数据查不到？**
A: 腾讯财经 API 需要能访问 `qt.gtimg.cn`。如果开了代理 (如 Clash)，确认该域名走直连或代理能通。

**Q: 知识库检索失败？**
A: 检查 `data/chroma_db/` 目录是否存在。删除后重启会自动重建。首次启动需下载嵌入模型 (~100MB)。

**Q: 想换其他 LLM？**
A: 修改 `.env` 中的 `DEEPSEEK_API_KEY`、`deepseek_base_url`、`model_name` 即可。任何 OpenAI 兼容接口都支持。

**Q: skeleton/mini/完整版 之间怎么切换？**
A: skeleton 是不完整的（需填空），mini 是能跑的单文件参考，完整版是模块化工程代码。建议按 skeleton→mini→完整 顺序学习。

---

## 📌 注意事项

1. ⚠️ 所有输出仅供参考，不构成投资建议
2. 📡 股票数据在非交易时间可能无最新价
3. 🔑 DeepSeek API 有调用配额限制
4. 💾 向量数据存储在 `./data/chroma_db/`，更换 Embedding 模型后需删除重建
5. 🐍 建议 Python 3.10+

---

## 🛠️ 技术栈

| 层 | 技术 |
|------|------|
| LLM | DeepSeek (OpenAI 兼容) |
| Agent 框架 | LangGraph |
| 工具框架 | LangChain `@tool` |
| 嵌入模型 | HuggingFace `BAAI/bge-small-zh-v1.5` |
| 向量数据库 | ChromaDB |
| 前端 | Gradio 6.x |
| API 服务 | FastAPI |
| 金融数据 | 腾讯财经 API + 天天基金 API |
| 配置 | pydantic-settings + python-dotenv |

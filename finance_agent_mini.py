"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           finance_agent_mini.py — 金融 AI Agent 超级缩略版                  ║
║           同架构 · 最少代码 · 完整流程 · 入门蓝图 · ~200 行                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

  架构流水线:
  ┌──────┐    ┌──────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────┐
  │ .env │ →  │Config│ →  │ LLM(DeepSeek)│ →  │Agent(ReAct)  │ →  │ Gradio  │
  └──────┘    └──────┘    │              │    │ +Tools+Memory│    │   UI    │
                          │ RAG(Chroma   │    └──────────────┘    └─────────┘
                          │ +HuggingFace)│
                          └──────────────┘

  快速启动:
    export DEEPSEEK_API_KEY=sk-xxx       # 或写入 .env 文件
    pip install langchain langchain-openai langchain-community \\
               langchain-chroma langchain-classic chromadb \\
               sentence-transformers gradio python-dotenv
    python finance_agent_mini.py          # 访问 http://localhost:7860
"""

import os
import uuid
from dotenv import load_dotenv
load_dotenv()

# ═══════════════════════════════════════════════════════════════════════════
# 第 1 层: 配置 (Config)
# ═══════════════════════════════════════════════════════════════════════════

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
MODEL_NAME = "deepseek-chat"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 本地中文嵌入模型，免费无 Key
CHROMA_DIR = "./data/chroma_mini"

# ═══════════════════════════════════════════════════════════════════════════
# 第 2 层: LLM + 嵌入模型
# ═══════════════════════════════════════════════════════════════════════════

from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.3,
    max_tokens=1024,
)

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

# ═══════════════════════════════════════════════════════════════════════════
# 第 3 层: 工具集 (Tools) — Agent 的手和眼
# ═══════════════════════════════════════════════════════════════════════════

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """
    获取当前日期时间，判断今天是否为A股交易日（周一至周五）。
    当用户询问"今天是什么日子"、"现在是几点"、"今天A股开盘吗"时使用。
    """
    from datetime import datetime
    now = datetime.now()
    weekday_name = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
    is_trading = now.weekday() < 5
    return (
        f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')} {weekday_name}\n"
        f"A股交易日判断: {'是交易日（仅排除周末）' if is_trading else '非交易日（周末）'}\n"
        f"A股交易时间: 9:30-11:30, 13:00-15:00"
    )


@tool
def compound_calculator(params: str = "") -> str:
    """
    复利计算器。传入一个 JSON 对象，包含 principal(本金/元)、rate(年利率%)、years(年数)。

    示例输入: {"principal": 100000, "rate": 8, "years": 10}
    计算逻辑: 最终金额 = 本金 × (1 + 年利率/100)^年数

    当用户询问"复利"、"多少年后多少钱"、"投资收益计算"时使用。
    """
    import json as _json
    try:
        p = _json.loads(params) if isinstance(params, str) else params
    except (_json.JSONDecodeError, TypeError):
        return f"参数格式错误: {params}，请提供 JSON，如 {{\"principal\":100000,\"rate\":8,\"years\":10}}"

    principal = float(p.get("principal", 0))
    rate = float(p.get("rate", 0))
    years = float(p.get("years", 1))
    result = principal * (1 + rate / 100) ** years
    profit = result - principal
    return (
        f"【复利计算结果】\n"
        f"  本金: {principal:,.0f} 元\n"
        f"  年利率: {rate}%\n"
        f"  投资年数: {years} 年\n"
        f"  最终金额: {result:,.2f} 元\n"
        f"  总收益: {profit:,.2f} 元 | 收益率: {profit / principal * 100:.1f}%"
    )


@tool
def search_knowledge(query: str) -> str:
    """
    搜索金融知识库，获取金融概念、投资策略、风险管理等专业知识。
    当用户询问"什么是XX"、"XX和XX有什么区别"、"XX策略"等知识性问题时使用。

    例如: query="什么是市盈率"
    """
    return _rag_query(query)


tools = [get_current_time, compound_calculator, search_knowledge]


# ═══════════════════════════════════════════════════════════════════════════
# 第 4 层: RAG 知识库 (向量存储 + 检索增强)
# ═══════════════════════════════════════════════════════════════════════════

from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# 初始化向量数据库
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="finance_mini",
)

# 预置金融知识（仅首次运行时入库，之后重启会跳过）
if not vectorstore.get()["ids"]:
    _knowledge_texts = [
        "市盈率(PE) = 股价 ÷ 每股收益(EPS)。PE 越低理论上估值越便宜，但需结合行业均值判断。"
        "一般 PE < 15 视为低估，15~30 为合理区间，> 30 为高估。适合横向对比同行业公司。",

        "复利公式: 终值 = 本金 × (1 + 年利率)^年数。"
        "例如 10 万元按 8% 年利率投资 10 年 ≈ 21.59 万元，总收益约 11.59 万。"
        "复利的核心是「时间」和「稳定收益率」，越早开始效果越好。",

        "价值投资核心理念: 以低于内在价值的价格买入优质公司，长期持有等待价值回归。"
        "代表人物: 本杰明·格雷厄姆（奠基人）、沃伦·巴菲特（集大成者）。"
        "常用指标: PE(市盈率)、PB(市净率)、ROE(净资产收益率)、自由现金流。",

        "止损策略之「2% 风险法则」: 单笔交易亏损不超过总资金的 2%。"
        "仓位计算公式: 最大仓位 = 总资产 × 2% ÷ 止损比例。"
        "例如总资产 50 万，止损设 5%，则最大仓位 = 500000×0.02÷0.05 = 20 万元。",

        "A 股交易规则: 交易日为周一至周五（法定节假日除外），交易时间上午 9:30-11:30，下午 13:00-15:00。"
        "实行 T+1 交收制度（当天买入次日才能卖出），涨跌幅限制主板 ±10%，创业板/科创板 ±20%。",
    ]
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    split_docs = splitter.split_documents(
        [Document(page_content=t) for t in _knowledge_texts]
    )
    vectorstore.add_documents(split_docs)


def _rag_query(question: str, k: int = 3) -> str:
    """
    RAG 核心流程: 检索 → 拼接上下文 → LLM 生成回答
    """
    # Step 1: 向量相似度检索
    docs = vectorstore.similarity_search(question, k=k)
    if not docs:
        return "（知识库中未检索到相关内容，请尝试换个问法。）"

    # Step 2: 拼接上下文
    context_parts = [
        f"[来源{i+1}] {doc.page_content}" for i, doc in enumerate(docs)
    ]
    context = "\n\n---\n\n".join(context_parts)

    # Step 3: LLM 根据上下文回答
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "你是金融分析师。仅根据以下【知识库内容】回答用户问题。\n"
            "若知识库无相关信息，请如实说明。回答简洁专业，50~200 字为宜。\n\n"
            "【知识库内容】\n{context}"
        )),
        ("human", "{question}"),
    ])
    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"context": context, "question": question})


# ═══════════════════════════════════════════════════════════════════════════
# 第 5 层: Agent (ReAct 推理循环 + 多轮记忆)
# ═══════════════════════════════════════════════════════════════════════════

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

# 会话记忆仓库: {session_id → ChatMessageHistory}
_session_store: dict[str, ChatMessageHistory] = {}


def _get_history(session_id: str) -> ChatMessageHistory:
    """为每个用户/会话维护独立的对话历史"""
    if session_id not in _session_store:
        _session_store[session_id] = ChatMessageHistory()
    return _session_store[session_id]


# ReAct 系统提示词（核心: 教会模型怎么思考→行动→观察→回答）
_SYSTEM_PROMPT = """你是一位专业的AI金融分析师助手，请用中文回答。

## 工作流程
收到用户问题后，按以下格式逐步处理（每次只使用一个工具）:

Thought: 分析用户需求，决定是否需要工具...
Action: 工具名称（必须是 {tool_names} 中的一个）
Action Input: 工具参数（JSON 格式）

工具返回结果后，你会看到 Observation。如果信息足够:
Thought: 我现在掌握了所有必要信息...
Final Answer: 用中文给出完整、专业的回答。

## 注意事项
- 涉及投资建议时，末尾必须加: "以上内容仅供参考，不构成投资建议。投资有风险，入市须谨慎。"
- 遇到计算类问题，先调用 compound_calculator 得到精确结果再回答。
- 遇到知识类问题，先调用 search_knowledge 搜索知识库再回答。"""


def create_agent() -> RunnableWithMessageHistory:
    """
    组装完整 Agent 流水线:
      LLM + Tools + ReAct Prompt → Agent → AgentExecutor → +Memory → 可对话 Agent
    """
    tool_names = [t.name for t in tools]

    # ReAct Prompt 模板（必须有 chat_history, input, agent_scratchpad, tools, tool_names）
    prompt = ChatPromptTemplate.from_messages([
        ("system",
            _SYSTEM_PROMPT
            + "\n\n## 可用工具\n\n{tools}\n\n## 工具名称列表: {tool_names}"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        ("assistant", "{agent_scratchpad}"),
    ])

    # 创建 ReAct Agent
    react_agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    # 包装为 Executor（控制最大轮次、自动重试解析错误）
    executor = AgentExecutor(
        agent=react_agent,
        tools=tools,
        verbose=False,              # True=打印详细推理过程
        max_iterations=6,            # 最多调用 6 次工具
        handle_parsing_errors=True,  # 格式错误自动重试
        return_intermediate_steps=False,
    )

    # 包装多轮记忆
    return RunnableWithMessageHistory(
        executor,
        _get_history,
        input_messages_key="input",
        history_messages_key="chat_history",
    )


# 全局单例 — 启动时创建，所有请求共享
agent = create_agent()


def chat(user_input: str, session_id: str = "default") -> str:
    """
    单次对话接口 — 所有调用方（Gradio UI / 命令行 / API）都通过此函数与 Agent 交互。
    """
    config = {"configurable": {"session_id": session_id}}
    try:
        result = agent.invoke({"input": user_input}, config=config)
        return result.get("output", "抱歉，未能获取回答，请重试。")
    except Exception as e:
        return f"处理请求时出错: {str(e)}"


# ═══════════════════════════════════════════════════════════════════════════
# 第 6 层: Gradio 聊天界面
# ═══════════════════════════════════════════════════════════════════════════

import gradio as gr


def _respond(message: str, history: list[dict]) -> str:
    """Gradio ChatInterface 回调 — 包装 chat() 函数"""
    return chat(message, session_id="gradio_user")


demo = gr.ChatInterface(
    fn=_respond,
    title="💹 财智助手 Mini — AI 金融分析师",
    description=(
        "基于 **LangChain + DeepSeek + RAG** 的金融智能助手（超级缩略学习版 ~200行）\n\n"
        "📊 可用工具: 时间查询 · 复利计算 · 知识库检索\n"
        "📚 内置知识: 市盈率 · 复利 · 价值投资 · 止损策略 · A股规则\n\n"
        "> ⚠️ 仅供学习参考，不构成投资建议。"
    ),
    examples=[
        "什么是市盈率？怎么判断估值高低？",
        "复利计算：本金10万元，年利率8%，投资10年",
        "价值投资是什么？代表人物有哪些？",
        "止损时的2%风险法则怎么算？",
        "现在几点？今天A股开盘吗？",
    ],
)

# ═══════════════════════════════════════════════════════════════════════════
# 第 7 层: 启动入口
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      💹 财智助手 Mini — 启动成功!                       ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  LLM:      {MODEL_NAME:<44s}║")
    print(f"║  Embedding:{EMBEDDING_MODEL:<44s}║")
    print(f"║  工具数:   {len(tools)} 个 ({', '.join(t.name for t in tools)})")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  访问:     http://localhost:7860                        ║")
    print("║  退出:     Ctrl+C                                       ║")
    print("╚══════════════════════════════════════════════════════════╝")
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)

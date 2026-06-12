"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     finance_agent_mini.py — AI 金融 Agent 完整参考实现 (~260 行)             ║
║     对应 skeleton 版的完整答案 · 每个 SECTION 都是可独立理解的组件             ║
╚══════════════════════════════════════════════════════════════════════════════╝

  架构一览:
  ┌──────────┐  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌──────────┐
  │ SEC 1    │  │ SEC 2    │  │ SEC 3+4       │  │ SEC 5    │  │ SEC 6    │
  │ Config   │→ │ LLM      │→ │ Tools + RAG   │→ │ Agent    │→ │ Gradio   │
  │ 环境变量 │  │ ChatOpenAI│  │ 3 工具+向量库 │  │ LangGraph│  │ UI       │
  └──────────┘  └──────────┘  └──────────────┘  └──────────┘  └──────────┘

  学习建议:
  - 先看 finance_agent_skeleton.py 尝试自己填空
  - 遇到卡住的步骤对照本文件找答案
  - 再去看完整项目 (app.py/agent.py/tools/) 了解生产级代码

  启动: python finance_agent_mini.py  →  http://localhost:7860
"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 1: 配置 ── 项目的"控制面板"                                     ║
# ║                                                                          ║
# ║  为什么这样设计？                                                         ║
# ║  - 集中管理: 所有变量一处修改，不用满项目找                                  ║
# ║  - 安全: API Key 通过环境变量传入，不会泄露到 git                            ║
# ║  - 可切换: 换模型只需改 MODEL_NAME，代码不动                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"  # OpenAI 兼容接口
MODEL_NAME = "deepseek-chat"
EMBEDDING_MODEL = "BAAI/bge-small-zh-v1.5"  # 中文嵌入模型, 本地运行, 免费
CHROMA_DIR = "./data/chroma_mini"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 2: LLM ── 大语言模型连接                                        ║
# ║                                                                          ║
# ║  ChatOpenAI 为什么能连 DeepSeek？                                         ║
# ║  → 因为 DeepSeek 提供了与 OpenAI 兼容的 API 格式                           ║
# ║  → 任何兼容 OpenAI 接口的服务商都可以用 ChatOpenAI 连接                     ║
# ║                                                                          ║
# ║  temperature=0.3: 金融场景要准确而非创意，所以偏低                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model=MODEL_NAME,
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
    temperature=0.3,
    max_tokens=1024,
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 3: 工具 ── Agent 的"手和眼"                                     ║
# ║                                                                          ║
# ║  @tool 装饰器做了什么？                                                   ║
# ║  1. 把普通函数包装成 LangChain Tool 对象                                   ║
# ║  2. 自动从函数签名 + docstring 生成工具描述 (给 LLM 看)                     ║
# ║  3. 处理参数验证和序列化                                                   ║
# ║                                                                          ║
# ║  工具设计原则:                                                            ║
# ║  - 单一职责: 一个工具只做一件事                                             ║
# ║  - 好 docstring: LLM 靠它判断何时调用、传什么参数                            ║
# ║  - 返回字符串: Agent 只处理文本，不要返回复杂对象                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from langchain_core.tools import tool


@tool
def get_current_time() -> str:
    """获取当前日期时间，判断是否 A 股交易日。用户问"现在几点"时使用。"""
    now = datetime.now()
    wd = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
    trading = "是交易日" if now.weekday() < 5 else "非交易日（周末）"
    return f"{now.strftime('%Y-%m-%d %H:%M:%S')} {wd} | A股: {trading} | 交易时间 9:30-15:00"


@tool
def compound_calculator(principal: float, rate: float, years: float) -> str:
    """
    复利计算器: 终值 = 本金 × (1 + 年利率/100)^年数。
    用户问"投资X元、利率Y%、Z年后"时调用。
    参数: principal-本金(元), rate-年利率(%), years-年数
    """
    result = principal * (1 + rate / 100) ** years
    return f"本金 {principal:,.0f}元 | 年利率 {rate}% | {years}年 → {result:,.2f}元 (收益 {result-principal:,.2f}元)"


@tool
def search_knowledge(query: str) -> str:
    """搜索金融知识库。用户问概念/策略时调用。参数: query-搜索词。"""
    return rag_query(query)


tools = [get_current_time, compound_calculator, search_knowledge]

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 4: RAG ── 检索增强生成                                          ║
# ║                                                                          ║
# ║  完整流程:                                                                ║
# ║  ① 文档 → TextSplitter(切小块) → Embedding(转向量) → ChromaDB(存储)       ║
# ║  ② 用户问题 → Embedding(转向量) → ChromaDB(相似度搜索) → 相关文档片段       ║
# ║  ③ 文档片段 + 问题 → Prompt → LLM → 回答                                  ║
# ║                                                                          ║
# ║  关键概念:                                                                ║
# ║  Embedding: 把文本变成数字向量。语义相近的文本 → 向量距离近                  ║
# ║  ChromaDB: 专门存向量的数据库，支持"找最相似的K条"这种操作                    ║
# ║  Chunk: 文档切分的小块。不能太大(超 token 限制)也不能太小(缺上下文)           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

embeddings = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
)

vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="finance_mini",
)

# 预置知识库 —— 只在这里定义，首次运行自动入库，重启跳过
FINANCE_KNOWLEDGE = [
    "市盈率(PE) = 股价 ÷ 每股收益。PE<15 低估、15-30 合理、>30 高估。不同行业差异大，银行股通常5-8倍，科技股可达50-100倍。须结合行业均值判断。",
    "复利: 终值 = 本金×(1+年利率)^年数。例: 10万按8%投资10年≈21.59万。核心要素是时间和稳定收益率，越早开始效果越好。年化收益率相差2%，30年后差距可达数倍。",
    "价值投资(格雷厄姆/巴菲特): 以低于内在价值的价格买入优质公司，长期持有。关注PE、PB、ROE、自由现金流，要求安全边际(30%-50%折扣)。成长投资: 关注高增长(营收增速>20%)，愿意接受高PE，关注PEG指标(PEG<1视为低估)。",
    "止损的2%法则: 单笔亏损不超过总资金2%。仓位 = 总资产×2%÷止损比例。例: 总资产50万、止损5%→最大仓位20万。这是风险管理的基本纪律。",
    "A股规则: 周一至周五交易(节假日除外)，9:30-11:30/13:00-15:00，T+1交收，主板±10%涨跌幅(创业板/科创板±20%，ST股±5%)。",
]

if not vectorstore.get()["ids"]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents([Document(page_content=t) for t in FINANCE_KNOWLEDGE])
    vectorstore.add_documents(chunks)


def rag_query(question: str, k: int = 3) -> str:
    """RAG 核心: 检索 → 拼接 → LLM 总结。仅检索不到时降级为直接返回原文。"""
    docs = vectorstore.similarity_search(question, k=k)
    if not docs:
        return "知识库中暂无相关信息。"
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "根据以下知识库内容简洁回答(50-200字):\n{context}"),
        ("human", "{question}"),
    ])
    try:
        return (prompt | llm | StrOutputParser()).invoke({"context": context, "question": question})
    except Exception:
        return context  # LLM 失败时直接返回检索原文

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 5: Agent ── 智能体核心                                          ║
# ║                                                                          ║
# ║  LangGraph create_react_agent 做了什么？                                   ║
# ║  1. 把 LLM + Tools + Prompt 组装成一个循环决策图                            ║
# ║  2. 使用模型的原生 function calling (不再需要手动解析文本格式)               ║
# ║  3. checkpointer=MemorySaver 让 Agent 记住之前的对话                        ║
# ║                                                                          ║
# ║  Agent 循环:                                                             ║
# ║  用户消息 → LLM(判断是否需要工具) → [需要]执行工具→返回结果给LLM→再判断      ║
# ║                                   → [不需要]生成最终回复 → 返回用户         ║
# ║                                                                          ║
# ║  关键概念:                                                                ║
# ║  checkpointer: 存对话状态(类似 HTTP session)。不用它的话每次对话都是全新的    ║
# ║  thread_id: 区分不同用户/会话的标识，类似数据库的主键                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()

SYSTEM_PROMPT = """你是专业的AI金融分析师助手，请用中文回答。
- 涉及投资建议时末尾加: "以上内容仅供参考，不构成投资建议。投资有风险，入市须谨慎。"
- 计算类问题先调 compound_calculator，知识类问题先调 search_knowledge。"""

agent = create_react_agent(
    model=llm,
    tools=tools,
    prompt=SYSTEM_PROMPT,
    checkpointer=memory,
)


def chat(user_input: str, session_id: str = "default") -> str:
    """单次对话。session_id 区分用户，同一 ID 共享对话历史。"""
    config = {"configurable": {"thread_id": session_id}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": user_input}]},
        config=config,
    )
    for msg in reversed(result.get("messages", [])):
        if hasattr(msg, "content") and msg.type == "ai" and msg.content:
            return msg.content
    return "抱歉，未能获取回答。"

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 6: UI ── Gradio 网页界面                                        ║
# ║                                                                          ║
# ║  ChatInterface vs Blocks:                                                ║
# ║  - ChatInterface: 最简单的聊天界面，适合快速原型。隐藏了 Chatbot/Textbox 等   ║
# ║  - Blocks: 完全自定义布局，可以加侧边栏、文件上传、多个 Chatbot 等             ║
# ║                                                                          ║
# ║  fn 参数: 接收 (message, history) → 返回回复字符串                          ║
# ║  examples: 页面下方的快捷提问按钮                                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import gradio as gr


def _respond(message: str, history: list[dict]) -> str:
    return chat(message, session_id="gradio_user")


demo = gr.ChatInterface(
    fn=_respond,
    title="💹 财智助手 Mini — AI 金融分析师",
    description=(
        "**LangChain + DeepSeek + RAG** 金融智能助手 (~260行参考实现)\n\n"
        "📊 工具: 时间查询 · 复利计算 · 知识库检索\n"
        "📚 内置知识: 市盈率 · 复利 · 价值投资 · 止损 · A股规则\n\n"
        "> 🧩 学习版: [skeleton](finance_agent_skeleton.py) → **mini**(本文件) → [完整项目](app.py)"
    ),
    examples=[
        "什么是市盈率？估值高低怎么判断？",
        "复利计算：本金10万，年利率8%，投10年",
        "价值投资和成长投资有什么区别？",
        "用2%法则帮我算：总资产50万，止损5%，仓位多少？",
        "现在几点？今天A股开盘吗？",
    ],
)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  SECTION 7: 启动                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════╗")
    print("║  💹 财智助手 Mini — 启动成功                         ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  LLM:  {MODEL_NAME:<42s}║")
    print(f"║  嵌入: {EMBEDDING_MODEL:<42s}║")
    print(f"║  工具: {len(tools)} 个 ({', '.join(t.name for t in tools)})")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  访问: http://localhost:7860                        ║")
    print("║  学习: skeleton → mini → 完整项目                    ║")
    print("╚══════════════════════════════════════════════════════╝")
    demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)

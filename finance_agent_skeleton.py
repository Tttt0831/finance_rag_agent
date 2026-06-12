"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          finance_agent_skeleton.py — AI Agent 渐进式填空学习版               ║
║                   从零搭建一个完整的金融 AI Agent                             ║
╚══════════════════════════════════════════════════════════════════════════════╝

  学习路径（由简到繁，每步可独立运行验证）:
  ┌─────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐
  │STEP0│ → │STEP1 │ → │STEP2 │ → │STEP3 │ → │STEP4 │ → │STEP5 │ → │STEP6 │
  │环境 │   │ 配置 │   │ LLM  │   │ 工具 │   │ RAG  │   │Agent │   │  UI  │
  └─────┘   └──────┘   └──────┘   └──────┘   └──────┘   └──────┘   └──────┘

  使用方法:
    1. 补齐每步的 TODO 代码
    2. 运行验证: python finance_agent_skeleton.py
    3. 对照 finance_agent_mini.py 查看完整答案
    4. 对照完整项目 (app.py / agent.py / tools/) 看生产级版本

  前置条件:
    pip install langchain langchain-openai langchain-community \
                langchain-chroma chromadb sentence-transformers \
                gradio python-dotenv requests langgraph
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 0: 导入库                                                          ║
# ║  导入所有需要的库，作为项目的基础依赖                                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件加载环境变量


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 1: 配置层 ── Config                                                ║
# ║  从环境变量读取 API Key 和模型配置。                                       ║
# ║  思考: 为什么要把配置集中管理？如果 Key 写死在代码里会有什么问题？           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# TODO 1.1: 读取 DeepSeek API Key
DEEPSEEK_API_KEY = "YOUR_API_KEY_HERE"  # ← 改成 os.getenv("DEEPSEEK_API_KEY", "")

# TODO 1.2: 设置 API 地址和模型名
DEEPSEEK_BASE_URL = ""  # ← 填入 DeepSeek 的 OpenAI 兼容地址
MODEL_NAME = ""          # ← 填入模型名，如 "deepseek-chat"

# TODO 1.3: 设置嵌入模型（用于 RAG 知识库）
EMBEDDING_MODEL = ""  # ← 填入 HuggingFace 中文嵌入模型名
CHROMA_DIR = "./data/chroma_skeleton"

# ── 验证 STEP 1 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"STEP 1 OK: API_KEY={'✅' if DEEPSEEK_API_KEY and 'sk-' in DEEPSEEK_API_KEY else '❌ 请填入 Key'}")
    print(f"           BASE_URL={DEEPSEEK_BASE_URL or '❌ 请填入'}")
    print(f"           MODEL={MODEL_NAME or '❌ 请填入'}")
    print(f"           EMBEDDING={EMBEDDING_MODEL or '❌ 请填入'}")
    print("\n👉 STEP 1 完成后继续 STEP 2\n")

# 取消下面注释以继续下一步:
# import sys; sys.exit(0)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 2: LLM 层 ── 大语言模型                                            ║
# ║  创建 ChatOpenAI 实例连接 DeepSeek。                                       ║
# ║  思考: ChatOpenAI 为什么能连接 DeepSeek？OpenAI 兼容接口是什么意思？        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from langchain_openai import ChatOpenAI

# TODO 2.1: 创建 LLM 实例
llm = None  # ← 用 ChatOpenAI(model=..., api_key=..., base_url=..., temperature=0.3, max_tokens=1024)

# 测试用 (取消注释以验证):
# response = llm.invoke("用一句话介绍你自己")
# print(f"LLM 回复: {response.content}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 3: 工具层 ── Tools                                                 ║
# ║  定义 Agent 可以调用的工具函数。@tool 装饰器会把函数变成 LangChain 工具。    ║
# ║  思考: 工具的 docstring 为什么重要？LLM 怎么知道该用哪个工具？               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from langchain_core.tools import tool


# TODO 3.1: 实现当前时间查询工具
@tool
def get_current_time() -> str:
    """
    获取当前日期时间。当用户询问"现在几点"、"今天是什么日子"时使用。

    注意: 这个 docstring 就是给 LLM 看的"工具使用说明书"。
    写得越清楚，LLM 越知道什么时候该调用它。
    """
    # TODO: 获取当前时间，返回包含日期、星期、是否是交易日的字符串
    return "请实现: 返回当前日期时间字符串"


# TODO 3.2: 实现复利计算工具
@tool
def compound_calculator(principal: float, rate: float, years: float) -> str:
    """
    复利计算器。当用户询问"投资X元、年利率Y%、Z年后多少钱"时使用。

    Args:
        principal: 本金（元）
        rate: 年利率（%），例如 8 表示 8%
        years: 投资年数
    """
    # TODO: 实现复利公式 终值 = 本金 × (1 + 年利率/100)^年数
    return "请实现: 复利计算"


# TODO 3.3: 实现知识库搜索工具（先留空，STEP 4 完成 RAG 后再回来填）
@tool
def search_knowledge(query: str) -> str:
    """
    搜索金融知识库，获取金融概念、投资策略等专业知识。
    当用户询问"什么是XX"、"XX策略"等知识性问题时使用。
    """
    # TODO: 调用 RAG 检索函数 (STEP 4 完成后实现)
    return "请在完成 STEP 4 后实现"


# ── 工具列表 ─────────────────────────────────────────────────────────────
tools = [get_current_time, compound_calculator, search_knowledge]

# ── 验证 STEP 3 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"STEP 3 OK: {len(tools)} 个工具已注册")
    for t in tools:
        print(f"  - {t.name}: {t.description[:50]}...")
    print(f"  get_current_time 测试: {get_current_time.invoke({})[:60]}")
    print(f"  compound_calculator 测试: {compound_calculator.invoke({'principal':100000,'rate':8,'years':10})[:60]}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 4: RAG 层 ── 检索增强生成                                         ║
# ║  向量数据库 + Embedding + 检索 → 让 Agent 能"读"文档                      ║
# ║                                                                          ║
# ║  数据流: 文档 → TextSplitter → Embedding → ChromaDB → 向量检索 → 结果    ║
# ║                                                                          ║
# ║  思考:                                                                   ║
# ║  1. Embedding 是什么？为什么文本不能直接存数据库？                          ║
# ║  2. ChromaDB 和传统数据库（如 MySQL）有什么不同？                           ║
# ║  3. 检索出来的结果是"原文"还是"AI 总结"？                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# TODO 4.1: 创建嵌入模型
embeddings = None  # ← 用 HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, ...)

# TODO 4.2: 创建向量数据库
vectorstore = None  # ← 用 Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings, ...)

# TODO 4.3: 定义预置金融知识（首次运行自动入库）
# 提示: 写 3-5 条金融知识点，每条 100-200 字
FINANCE_KNOWLEDGE = [
    "",  # 知识点 1: 市盈率 PE
    "",  # 知识点 2: 复利
    "",  # 知识点 3: 价值投资
]


# TODO 4.4: 首次运行时将知识入库
# 提示: 1. 检查 vectorstore 是否为空  2. 用 TextSplitter 分割  3. 添加到 vectorstore
def _init_knowledge():
    """如果向量库为空，则将 FINANCE_KNOWLEDGE 分割后存入。"""
    pass  # ← 实现此函数


# TODO 4.5: 实现 RAG 检索函数
def rag_query(question: str, k: int = 3) -> str:
    """
    完整 RAG 流程: 向量检索 → 拼接上下文 → LLM 总结回答

    这是整个 RAG 的核心，理解了这个函数就理解了 RAG:
    1. vectorstore.similarity_search() — 向量相似度检索
    2. 拼接检索到的文档片段为上下文
    3. 将上下文 + 问题发给 LLM，让它基于上下文回答
    """
    # 步骤 1: 检索相关文档
    # docs = vectorstore.similarity_search(question, k=k)

    # 步骤 2: 拼接上下文
    # context = "\n\n".join(d.page_content for d in docs)

    # 步骤 3: 构建 Prompt 并调用 LLM
    # prompt = ChatPromptTemplate.from_messages([...])
    # chain = prompt | llm | StrOutputParser()
    # return chain.invoke({"context": context, "question": question})

    return "请实现 RAG 检索函数"


# ── 验证 STEP 4 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    _init_knowledge()
    result = rag_query("什么是市盈率")
    print(f"STEP 4 RAG 测试: {result[:200] if len(result)>10 else '❌ 请实现 rag_query'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 5: Agent 层 ── LangGraph Agent                                     ║
# ║  把 LLM + Tools + Memory 组装成一个能自主推理的 Agent。                     ║
# ║                                                                          ║
# ║  Agent 的工作流程:                                                        ║
# ║  用户输入 → LLM 判断是否需要工具 → [需要] 调用工具 → 观察结果 → LLM 再判断  ║
# ║                                    → [不需要] 直接回答                    ║
# ║                                                                          ║
# ║  思考:                                                                   ║
# ║  1. Agent 和普通 ChatBot 有什么区别？                                      ║
# ║  2. LangGraph 的 checkpointer 是干什么的？                                 ║
# ║  3. 如果不加 checkpointer，Agent 还能记住之前的对话吗？                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver

# 全局记忆（跨轮对话共享上下文）
memory = MemorySaver()

# 系统提示词 —— 定义 Agent 的身份和行为规则
SYSTEM_PROMPT = """你是一位专业的AI金融分析师助手，请用中文回答。

注意事项:
- 涉及投资建议时，末尾必须加: "以上内容仅供参考，不构成投资建议。投资有风险，入市须谨慎。"
- 遇到计算类问题，先调用 compound_calculator 得到精确结果再回答。
- 遇到知识类问题，先调用 search_knowledge 搜索知识库再回答。"""


# TODO 5.1: 创建 Agent
# 提示: create_react_agent(model=llm, tools=tools, prompt=SYSTEM_PROMPT, checkpointer=memory)
agent = None  # ← 创建 Agent


# TODO 5.2: 实现对话函数
def chat(user_input: str, session_id: str = "default") -> str:
    """
    单次对话接口。session_id 用于区分不同的用户/会话。

    LangGraph Agent 的调用方式:
      agent.invoke({"messages": [{"role": "user", "content": user_input}]},
                   config={"configurable": {"thread_id": session_id}})

    返回结果中, "messages" 列表包含所有历史消息, 最后一条 type=="ai" 的消息就是回复。
    """
    # TODO: 调用 agent.invoke() 并提取最后一条 AI 消息
    return "请实现 Agent 对话函数"


# ── 验证 STEP 5 ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    _init_knowledge()
    response = chat("用一句话介绍你自己")
    print(f"STEP 5 Agent 测试: {response[:200] if len(response)>10 else '❌ 请实现 Agent'}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  STEP 6: UI 层 ── Gradio 聊天界面                                        ║
# ║  把 Agent 包装成一个网页聊天界面，用户可以在浏览器中使用。                    ║
# ║                                                                          ║
# ║  思考:                                                                   ║
# ║  1. ChatInterface 和 Blocks 的区别是什么？什么时候用哪个？                  ║
# ║  2. 为什么 fn 参数要包装 chat() 函数？不包装会怎样？                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

import gradio as gr


def _respond(message: str, history: list[dict]) -> str:
    """Gradio 回调 — 接收用户消息，返回 Agent 回复"""
    return chat(message, session_id="gradio_user")


# TODO 6.1: 创建 Gradio 聊天界面
# 提示: gr.ChatInterface(fn=_respond, title="...", description="...", examples=[...])
demo = None  # ← 创建 ChatInterface


# ── 启动 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    _init_knowledge()
    print("╔════════════════════════════════════════════════╗")
    print("║  🎉 恭喜完成所有步骤！                          ║")
    print("║  访问 http://localhost:7860                     ║")
    print("║  对照 finance_agent_mini.py 查看完整答案        ║")
    print("╚════════════════════════════════════════════════╝")
    if demo:
        demo.launch(server_name="0.0.0.0", server_port=7860, inbrowser=True)
    else:
        print("\n❌ 请先完成 STEP 6: 创建 ChatInterface")

"""
app.py - 金融 Agent 前端应用（Gradio）
功能：
  - 多轮对话聊天界面
  - 文档上传入库（支持 PDF / TXT / CSV / Excel）
  - 实时展示 Agent 思考过程（可选）
  - 快捷示例问题
"""

import os
import uuid
import gradio as gr

from agent import create_finance_agent, get_agent_response
from services.vector_store import vector_store_service
from utils.logger import get_logger

logger = get_logger(__name__)

# ── 初始化 Agent（全局单例）────────────────────────────────────
logger.info("应用启动，初始化金融 Agent...")
_agent = create_finance_agent()
logger.info("金融 Agent 初始化完成，等待用户连接。")

# ── 预加载内置知识库 ─────────────────────────────────────────
def _preload_knowledge():
    """将内置金融知识文档自动加载到向量库。"""
    knowledge_file = os.path.join(
        os.path.dirname(__file__), "data", "finance_knowledge.txt"
    )
    if os.path.exists(knowledge_file):
        result = vector_store_service.add_file(knowledge_file)
        if result["status"] == "success":
            logger.info(f"内置知识库加载成功: {result['chunks']} 个知识块")
        else:
            logger.info("内置知识库已存在，跳过加载")

_preload_knowledge()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 聊天处理函数
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chat(
    user_message: str,
    history: list,
    session_id: str,
) -> tuple[str, list]:
    """
    处理用户消息，返回 Agent 回复。

    Args:
        user_message: 用户输入
        history: Gradio 格式的历史消息列表
        session_id: 当前会话 ID

    Returns:
        (空字符串清空输入框, 更新后的历史)
    """
    if not user_message.strip():
        return "", history

    logger.info(f"[用户({session_id[:8]})] {user_message}")

    # 调用 Agent
    response = get_agent_response(_agent, user_message, session_id)

    logger.info(f"[Agent 回复] {response[:100]}...")
    # Gradio 6 Chatbot 使用 dict 格式
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    return "", history


def upload_document(files, status_text: str) -> str:
    """
    处理用户上传的文档，将其入库到向量数据库。

    Args:
        files: Gradio 上传的文件列表
        status_text: 当前状态文本

    Returns:
        更新后的状态文本
    """
    if not files:
        return "⚠️ 请先选择文件"

    results = []
    for file in files:
        try:
            result = vector_store_service.add_file(file.name)
            filename = os.path.basename(file.name)
            if result["status"] == "success":
                results.append(f"✅ {filename}：成功入库 {result['chunks']} 个知识块")
            else:
                results.append(f"⏭️ {filename}：已存在，跳过")
        except Exception as e:
            filename = os.path.basename(file.name) if file else "未知文件"
            results.append(f"❌ {filename}：入库失败 - {str(e)}")
            logger.error(f"文件入库失败: {e}")

    return "\n".join(results)


def generate_session_id() -> str:
    """生成新的会话 ID。"""
    return str(uuid.uuid4())


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Gradio 界面构建
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXAMPLE_QUESTIONS = [
    "今天大盘行情怎么样？",
    "帮我查一下贵州茅台（600519）的实时股价",
    "宁德时代（300750）最近20天的走势如何？",
    "什么是市盈率？怎么用它判断股票估值？",
    "我有10万元，每年预期收益8%，10年后能有多少钱？",
    "价值投资和成长投资有什么区别？",
    "帮我分析一下平安银行（000001）的基本面财务指标",
    "如果我用5万买了一只股票，止损设在5%，我最多亏多少？",
    "现在适合定投沪深300指数基金吗？",
]

INTRODUCTION = """
## 💹 财智助手 - AI 金融分析师

我是基于 **LangChain + RAG + Agent** 技术构建的金融智能助手，可以帮您：

- 📊 **查询行情**：A股实时股价、指数涨跌、历史K线
- 📈 **基本面分析**：PE/PB/市值等财务指标
- 📚 **知识问答**：金融概念、投资策略、风险管理
- 🧮 **金融计算**：复利、收益率、仓位大小
- 📁 **私有知识库**：上传您的研报、财报，我来分析

> ⚠️ 本工具仅提供信息参考，不构成投资建议。股市有风险，投资须谨慎。
"""

with gr.Blocks(
    title="财智助手 - AI 金融分析师",
) as demo:

    # 会话 ID（每次刷新页面生成新的）
    session_id = gr.State(generate_session_id)

    # ── 头部 ─────────────────────────────────────────────────
    gr.Markdown(INTRODUCTION)
    gr.Markdown("---")

    with gr.Row():
        # ── 左侧：聊天区 ─────────────────────────────────────
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="对话",
                height=520,
                avatar_images=(
                    None,  # 用户头像
                    "https://api.dicebear.com/7.x/bottts/svg?seed=finance",  # AI头像
                ),
            )

            with gr.Row():
                user_input = gr.Textbox(
                    placeholder="请输入您的金融问题，如：帮我查茅台今天的股价...",
                    label="",
                    lines=1,
                    scale=5,
                    show_label=False,
                )
                submit_btn = gr.Button("发送 ▶", variant="primary", scale=1)

            with gr.Row():
                clear_btn = gr.Button("🗑️ 清空对话", size="sm")
                new_session_btn = gr.Button("🔄 新建会话", size="sm")

            # ── 示例问题 ──────────────────────────────────────
            gr.Markdown("**💡 快速提问示例：**")
            with gr.Row():
                for q in EXAMPLE_QUESTIONS[:4]:
                    gr.Button(q, size="sm", elem_classes="example-btn").click(
                        fn=lambda x=q: x,
                        outputs=user_input,
                    )
            with gr.Row():
                for q in EXAMPLE_QUESTIONS[4:]:
                    gr.Button(q, size="sm", elem_classes="example-btn").click(
                        fn=lambda x=q: x,
                        outputs=user_input,
                    )

        # ── 右侧：文档上传 + 说明 ─────────────────────────────
        with gr.Column(scale=1):
            gr.Markdown("### 📁 上传私有文档")
            gr.Markdown(
                "将您的研报、财报、公告等文档上传后，我可以基于其内容回答问题。\n\n"
                "支持格式：PDF、TXT、Markdown、CSV、Excel"
            )

            file_upload = gr.File(
                label="选择文件（可多选）",
                file_count="multiple",
                file_types=[".pdf", ".txt", ".md", ".csv", ".xlsx", ".xls"],
            )
            upload_btn = gr.Button("📤 上传并入库", variant="secondary")
            upload_status = gr.Textbox(
                label="上传状态",
                interactive=False,
                lines=4,
                placeholder="上传结果将显示在这里...",
            )

            gr.Markdown("---")

            gr.Markdown("### 🛠️ 可用工具")
            gr.Markdown("""
- 🔍 **知识库检索** - 金融概念/策略
- 📊 **实时股价** - A股最新行情
- 📉 **历史K线** - 近期趋势数据
- 💹 **财务指标** - PE/PB/市值
- 📈 **市场指数** - 沪深300等主要指数
- 💰 **基金净值** - 公募基金数据
- 🧮 **金融计算器** - 复利/收益率/仓位
- 🕐 **时间查询** - 交易日判断
""")

    # ── 事件绑定 ──────────────────────────────────────────────

    # 发送消息（点击按钮）
    submit_btn.click(
        fn=chat,
        inputs=[user_input, chatbot, session_id],
        outputs=[user_input, chatbot],
        queue=True,
    )

    # 发送消息（回车）
    user_input.submit(
        fn=chat,
        inputs=[user_input, chatbot, session_id],
        outputs=[user_input, chatbot],
        queue=True,
    )

    # 清空对话
    clear_btn.click(fn=lambda: [], outputs=chatbot)

    # 新建会话（重置 session_id，历史记忆清空）
    new_session_btn.click(
        fn=lambda: ([], generate_session_id()),
        outputs=[chatbot, session_id],
    )

    # 文件上传
    upload_btn.click(
        fn=upload_document,
        inputs=[file_upload, upload_status],
        outputs=upload_status,
    )


# ── 启动 ────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        inbrowser=True,   # 自动在浏览器中打开
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="slate",
        ),
        css="""
        .message.svelte-1lcyrx4 { font-size: 15px; }
        .example-btn { margin: 3px !important; }
        footer { display: none !important; }
        """,
    )

"""
eval/evaluate.py - RAG 评测（金标集 + 分段指标）

对照《Agent 开发知识库》模块 03/04：
  - 检索段：Hit@k / Recall@k（关键词覆盖）/ MRR（首个命中排名倒数）
  - 生成段（可选 --generate）：LLM-as-judge 判 Faithfulness（答案是否被上下文支撑）

检索段完全离线可跑（无需 DeepSeek API Key）；生成段需配置 DEEPSEEK_API_KEY。

用法：
  python eval/evaluate.py                  # 混合检索，k=4
  python eval/evaluate.py --dense-only     # 纯稠密检索，做 A/B 对比
  python eval/evaluate.py --k 6            # 调 top-k
  python eval/evaluate.py --generate       # 追加生成段 Faithfulness 评测
"""

import argparse
import json
import os
import sys

# 允许从项目根目录直接运行
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import settings
from services.vector_store import vector_store_service
from services.hybrid_retriever import hybrid_retriever
from utils.logger import get_logger

logger = get_logger(__name__)

_BUILTIN_KB = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "finance_knowledge.txt",
)

# 金标集路径：相对本脚本定位，避免依赖运行目录
_DEFAULT_GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_set.json")


def ensure_indexed() -> int:
    """确保内置知识库已入库（评测不应依赖 app.py 的预加载）。"""
    docs = vector_store_service.get_all_documents()
    if docs:
        return len(docs)
    if os.path.exists(_BUILTIN_KB):
        vector_store_service.add_file(_BUILTIN_KB)
    return len(vector_store_service.get_all_documents())


def load_golden(path: str) -> list:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieval_metrics(question: str, keywords: list, k: int, use_hybrid: bool) -> dict:
    """对单条问题计算 Hit@k / Recall@k / first-hit rank。"""
    docs = hybrid_retriever.search(question, k=k, use_hybrid=use_hybrid)
    text = "\n".join(d.page_content for d in docs)

    hit = any(kw in text for kw in keywords)
    covered = sum(1 for kw in keywords if kw in text)
    recall = covered / len(keywords) if keywords else 0.0

    first_rank = 0
    for i, d in enumerate(docs, start=1):
        if any(kw in d.page_content for kw in keywords):
            first_rank = i
            break

    return {"hit": hit, "recall": recall, "first_rank": first_rank,
            "n_docs": len(docs), "covered": covered, "total_kw": len(keywords)}


def run_retrieval(golden: list, k: int, use_hybrid: bool) -> dict:
    rows = []
    hits = recall_sum = mrr_sum = 0
    for item in golden:
        r = retrieval_metrics(item["question"], item["gold_keywords"], k, use_hybrid)
        mrr = (1.0 / r["first_rank"]) if r["first_rank"] else 0.0
        hits += int(r["hit"])
        recall_sum += r["recall"]
        mrr_sum += mrr
        rows.append({**item, **r, "mrr": round(mrr, 4)})

    n = len(golden)
    return {
        "k": k,
        "method": "hybrid" if use_hybrid else "dense",
        "n": n,
        "hit@k": round(hits / n, 4) if n else 0.0,
        "recall@k": round(recall_sum / n, 4) if n else 0.0,
        "mrr": round(mrr_sum / n, 4) if n else 0.0,
        "rows": rows,
    }


def run_generation(golden: list, k: int) -> dict:
    """LLM-as-judge 判 Faithfulness（需 API Key，失败时返回空）。"""
    if not settings.deepseek_api_key:
        logger.warning("未配置 DEEPSEEK_API_KEY，跳过生成段评测")
        return {}
    try:
        from langchain_openai import ChatOpenAI
        from services.rag_service import _format_docs
        llm = ChatOpenAI(
            model=settings.model_name, api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
            temperature=0.0, max_tokens=64,
        )
    except Exception as e:
        logger.warning(f"生成段评测初始化失败，跳过: {e}")
        return {}

    scores = []
    for item in golden:
        q = item["question"]
        docs = hybrid_retriever.search(q, k=k, use_hybrid=settings.enable_hybrid)
        answer = _answer(q, docs, llm)
        score = _judge(q, _format_docs(docs), answer, llm)
        if score is not None:
            scores.append(score)
        logger.info(f"[faithfulness] {q[:20]}... → {score}")
    if not scores:
        return {}
    return {"faithfulness": round(sum(scores) / len(scores), 3), "n": len(scores)}


def _answer(question: str, docs: list, llm) -> str:
    from services.rag_service import _format_docs
    try:
        return str(llm.invoke(
            f"根据资料回答（资料不足请说明）：\n{_format_docs(docs)}\n\n问题：{question}"
        ).content)
    except Exception:
        return ""


def _judge(question: str, context: str, answer: str, llm):
    if not answer:
        return None
    prompt = (
        "你是严谨的评测员。判断下面的【回答】是否被【资料】支撑（Faithfulness）。\n"
        "只输出 1-5 的整数：1=完全无支撑/编造，3=部分支撑，5=完全被支撑。\n\n"
        f"【问题】{question}\n【资料】{context[:2000]}\n【回答】{answer[:1000]}"
    )
    try:
        out = str(llm.invoke(prompt).content).strip()
        s = int(out[0]) if out and out[0].isdigit() else 3
        return min(max(s, 1), 5)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="RAG 金标集评测")
    ap.add_argument("--k", type=int, default=settings.retriever_k, help="检索 top-k")
    ap.add_argument("--dense-only", action="store_true", help="纯稠密检索 A/B 对比")
    ap.add_argument("--generate", action="store_true", help="追加生成段 Faithfulness 评测")
    ap.add_argument("--golden", default=_DEFAULT_GOLDEN, help="金标集路径")
    args = ap.parse_args()

    n_docs = ensure_indexed()
    logger.info(f"向量库已就绪：{n_docs} 块")

    golden = load_golden(args.golden)
    if not golden:
        logger.error("金标集为空")
        sys.exit(1)

    use_hybrid = not args.dense_only
    result = run_retrieval(golden, args.k, use_hybrid)

    # 控制台表格
    print(f"\n{'=' * 60}")
    print(f"RAG 评测 · {result['method']} · k={result['k']} · {result['n']} 条")
    print(f"{'=' * 60}")
    print(f"{'id':>3}  {'hit':>5}  {'recall':>6}  {'mrr':>6}  question")
    for r in result["rows"]:
        mark = "✓" if r["hit"] else "✗"
        print(f"{r['id']:>3}  {mark:>5}  {r['recall']:>6.2f}  {r['mrr']:>6.2f}  {r['question'][:30]}")
    print(f"{'-' * 60}")
    print(f"Hit@{result['k']}   = {result['hit@k']:.2%}")
    print(f"Recall@{result['k']} = {result['recall@k']:.2%}")
    print(f"MRR       = {result['mrr']:.2%}")

    report = {"retrieval": {k: v for k, v in result.items() if k != "rows"}}

    if args.generate:
        gen = run_generation(golden, args.k)
        if gen:
            print(f"Faithfulness = {gen['faithfulness']}/5  (n={gen['n']})")
            report["generation"] = gen

    # 落盘 report.json
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n报告已写入: {out_path}")


if __name__ == "__main__":
    main()

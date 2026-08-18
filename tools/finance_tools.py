"""
tools/finance_tools.py - 金融 Agent 工具集（腾讯财经 API 版）
  1. search_knowledge_base  - 检索金融知识库（RAG）
  2. get_stock_realtime     - A 股实时行情
  3. get_stock_history      - 历史 K 线数据
  4. get_stock_financial    - 财务指标（PE/PB/市值）
  5. get_market_index       - 主要市场指数
  6. get_fund_info          - 基金净值
  7. calculate_financial    - 金融计算器
  8. get_current_datetime   - 当前日期时间
"""

import json
from datetime import datetime

from langchain_core.tools import tool

from services.rag_service import rag_service
from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════════

def _str(v):
    """兼容 LangGraph 可能传入的 dict 格式"""
    if isinstance(v, dict):
        return v.get("query") or v.get("input") or next(iter(v.values()), str(v))
    return str(v) if v else ""


def _n(val, precision: int = 2):
    """安全数字格式化"""
    if val is None or val == "" or val == "-":
        return "N/A"
    try:
        return round(float(val), precision)
    except (ValueError, TypeError):
        return str(val)


# 腾讯财经 qt.gtimg.cn 返回的 ~ 分隔字段下标（具名常量，避免魔法数字）
# 形如: v_sh600519="1~贵州茅台~600519~1689.00~...";  字段顺序固定，下标含义如下：
_TX_FIELD = {
    "name": 1,            # 股票名称
    "price": 3,           # 最新价
    "prev_close": 4,      # 昨收
    "open": 5,            # 今开
    "volume": 6,          # 成交量(手)
    "change": 31,         # 涨跌额
    "change_pct": 32,     # 涨跌幅(%)
    "high": 33,           # 最高
    "low": 34,            # 最低
    "turnover": 38,       # 换手率(%)
    "pe": 39,             # 市盈率(动)
    "market_cap": 45,     # 总市值
    "circulation_cap": 46,  # 流通市值
}
# 解析所需的最大下标，字段数不足则视为数据异常
_TX_MIN_FIELDS = max(_TX_FIELD.values()) + 1


def _tx_get(codes: list) -> dict:
    """
    调用腾讯财经 API (qt.gtimg.cn)，经用户代理稳定连接。
    codes: ["sh600519", "sz000001", "s_sh000001"]
    返回 {code: {...}}
    """
    import requests
    url = f"https://qt.gtimg.cn/q={','.join(codes)}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"}
    r = requests.get(url, headers=headers, timeout=8)
    r.raise_for_status()
    r.encoding = "gbk"  # 腾讯接口返回 GBK 编码，否则中文名乱码
    results = {}
    for line in r.text.strip().split("\n"):
        line = line.strip()
        if not line or '="' not in line:
            continue
        name_part, data_part = line.split('="', 1)
        code = name_part.split("_")[-1] if "_" in name_part else name_part[2:]
        fields = data_part.rstrip('";').split("~")
        if len(fields) < _TX_MIN_FIELDS:
            # 字段数不足：可能是接口改版或代码无效，跳过并记录便于排查
            logger.warning(
                f"[腾讯API] {code} 字段数异常({len(fields)} < {_TX_MIN_FIELDS})，已跳过"
            )
            continue
        results[code] = {key: fields[idx] for key, idx in _TX_FIELD.items()}
    return results


def _tx_get_index(codes: list) -> dict:
    """
    获取指数的「简版行情」(s_ 前缀)。腾讯简版只返回约 8~12 个字段，
    布局与个股完整行情不同：[1]=名称 [3]=最新价 [4]=涨跌额 [5]=涨跌幅%。
    返回 {code: {"name","price","change","change_pct"}}
    """
    import requests
    url = f"https://qt.gtimg.cn/q={','.join(codes)}"
    headers = {"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"}
    r = requests.get(url, headers=headers, timeout=8)
    r.raise_for_status()
    r.encoding = "gbk"
    results = {}
    for line in r.text.strip().split("\n"):
        line = line.strip()
        if not line or '="' not in line:
            continue
        name_part, data_part = line.split('="', 1)
        code = name_part.split("v_")[-1]  # 如 v_s_sh000001 -> s_sh000001
        fields = data_part.rstrip('";').split("~")
        if len(fields) < 6:
            logger.warning(f"[腾讯API-指数] {code} 字段数异常({len(fields)})，已跳过")
            continue
        results[code] = {
            "name": fields[1], "price": fields[3],
            "change": fields[4], "change_pct": fields[5],
        }
    return results


def _to_tx(code: str) -> str:
    """6 位代码 → 腾讯格式"""
    if code.startswith(("6", "5", "9")):
        return f"sh{code}"
    return f"sz{code}"


# ═══════════════════════════════════════════════════════════════
# 工具 1：金融知识库检索（RAG）
# ═══════════════════════════════════════════════════════════════

@tool
def search_knowledge_base(query: str = "") -> str:
    """
    检索金融知识库，获取专业金融知识、投资策略等。
    Args: query: 搜索关键词，如"市盈率"
    """
    query = _str(query)
    logger.info(f"[工具] search_knowledge_base: {query}")
    try:
        return rag_service.query(str(query))
    except Exception as e:
        logger.error(f"RAG LLM检索失败，降级为直接向量搜索: {e}")
        # 降级方案：跳过 LLM 总结，直接返回原始文档片段
        try:
            from services.hybrid_retriever import hybrid_retriever
            docs = hybrid_retriever.search(str(query), k=3)
            if docs:
                parts = [f"[来源] {d.metadata.get('source_file','知识库')}\n{d.page_content}" for d in docs]
                return "（直接检索结果，未经 LLM 总结）\n\n" + "\n\n---\n\n".join(parts)
            return f"知识库中暂无与「{query}」相关的内容。"
        except Exception as e2:
            return f"知识库检索失败: {str(e2)[:100]}"


# ═══════════════════════════════════════════════════════════════
# 工具 2：A 股实时行情
# ═══════════════════════════════════════════════════════════════

@tool
def get_stock_realtime(stock_code: str = "") -> str:
    """
    获取 A 股实时行情（最新价、涨跌幅、PE、市值）。
    stock_code: 6 位代码，如 "600519"（茅台）、"000001"（平安银行）
    """
    stock_code = _str(stock_code)
    logger.info(f"[工具] get_stock_realtime: {stock_code}")
    try:
        data = _tx_get([_to_tx(stock_code)])
        if not data:
            return f"未找到股票 {stock_code}。"
        d = list(data.values())[0]
        result = {
            "股票代码": stock_code, "股票名称": d["name"],
            "最新价": _n(d["price"]), "昨收": _n(d["prev_close"]),
            "涨跌幅(%)": _n(d["change_pct"]), "涨跌额": _n(d["change"]),
            "今开": _n(d["open"]), "最高": _n(d["high"]), "最低": _n(d["low"]),
            "成交量(手)": _n(d["volume"]), "市盈率": _n(d["pe"]),
            "总市值": _n(d["market_cap"]), "流通市值": _n(d["circulation_cap"]),
            "数据时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)
    except Exception as e:
        return f"获取行情失败（网络原因）: {str(e)[:100]}"


# ═══════════════════════════════════════════════════════════════
# 工具 3：股票历史 K 线
# ═══════════════════════════════════════════════════════════════

@tool
def get_stock_history(stock_code: str = "", period: str = "daily", limit: int = 20) -> str:
    """
    获取 A 股历史 K 线（日线/周线/月线）。
    Args:
        stock_code: 6 位代码, period: daily/weekly/monthly, limit: 最多 40 条
    """
    stock_code = _str(stock_code)
    klt_map = {"daily": "day", "weekly": "week", "monthly": "month", "日线": "day", "周线": "week", "月线": "month"}
    klt = klt_map.get(period, "day")
    limit = min(int(limit), 40)
    logger.info(f"[工具] get_stock_history: {stock_code}, klt={klt}, limit={limit}")
    try:
        import requests
        tx_code = _to_tx(stock_code)
        url = f"https://web.ifzq.gtimg.cn/appstock/app/fqkline/get?param={tx_code},{klt},,,{limit},qfq"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.qq.com/"}, timeout=8)
        r.raise_for_status()
        klines = r.json().get("data", {}).get(tx_code, {}).get(f"qfq{klt}", []) or []
        if not klines:
            return f"股票 {stock_code} 暂无 K 线数据。"
        closes = [float(k[2]) for k in klines]
        lines = [f"{k[0]}  开{k[1]} 收{k[2]} 高{k[3]} 低{k[4]} 量{k[5]}" for k in klines[-10:]]
        summary = {
            "股票代码": stock_code, "数据条数": len(klines),
            "最高": round(max(closes), 2), "最低": round(min(closes), 2),
            "最新收盘": closes[-1],
            "区间涨跌幅(%)": round((closes[-1]-closes[0])/closes[0]*100, 2),
        }
        return json.dumps(summary, ensure_ascii=False, indent=2) + "\n\n" + "\n".join(lines)
    except Exception as e:
        return f"获取历史数据失败: {str(e)[:100]}"


# ═══════════════════════════════════════════════════════════════
# 工具 4：股票财务指标
# ═══════════════════════════════════════════════════════════════

@tool
def get_stock_financial(stock_code: str = "") -> str:
    """
    获取核心财务指标（PE、市值、换手率等），用于基本面分析。
    """
    stock_code = _str(stock_code)
    logger.info(f"[工具] get_stock_financial: {stock_code}")
    try:
        data = _tx_get([_to_tx(stock_code)])
        if not data:
            return f"未找到 {stock_code} 的数据。"
        d = list(data.values())[0]
        result = {
            "股票代码": stock_code, "股票名称": d["name"],
            "最新价": _n(d["price"]), "涨跌幅(%)": _n(d["change_pct"]),
            "市盈率(动)": _n(d["pe"]),
            "总市值": _n(d["market_cap"]), "流通市值": _n(d["circulation_cap"]),
            "换手率(%)": _n(d.get("turnover")),
        }
        return json.dumps(result, ensure_ascii=False, indent=2) + "\n\n⚠️ 仅供参考，不构成投资建议。"
    except Exception as e:
        return f"获取财务指标失败: {str(e)[:100]}"


# ═══════════════════════════════════════════════════════════════
# 工具 5：市场指数行情
# ═══════════════════════════════════════════════════════════════

_INDICES = {
    "上证指数": "s_sh000001", "深证成指": "s_sz399001",
    "创业板指": "s_sz399006", "科创50": "s_sh000688",
    "沪深300": "s_sh000300", "上证50": "s_sh000016",
}

@tool
def get_market_index() -> str:
    """
    获取主要 A 股指数实时行情：上证、深证、创业板、科创50、沪深300、上证50。
    """
    logger.info("[工具] get_market_index")
    try:
        data = _tx_get_index(list(_INDICES.values()))
        if not data:
            return "暂时无法获取指数行情。"
        now = datetime.now().strftime("%Y-%m-%d %H:%M")
        out = f"【主要指数行情 {now}】\n\n"
        for name, code in _INDICES.items():
            d = data.get(code)
            if d:
                out += f"  {name}: {_n(d['price'])}  ({_n(d['change_pct'])}%)\n"
        out += "\n⚠️ 数据可能延迟约 15 秒，仅供参考。"
        return out
    except Exception as e:
        return f"获取指数行情失败: {str(e)[:100]}"


# ═══════════════════════════════════════════════════════════════
# 工具 6：基金净值查询
# ═══════════════════════════════════════════════════════════════

@tool
def get_fund_info(fund_code: str = "") -> str:
    """
    获取公募基金最新净值。fund_code: 6 位代码，如 "110022"（易方达消费）。
    """
    fund_code = _str(fund_code)
    logger.info(f"[工具] get_fund_info: {fund_code}")
    try:
        import requests
        url = f"https://fundgz.1234567.com.cn/js/{fund_code}.js"
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0", "Referer": "https://fund.eastmoney.com/"}, timeout=8)
        r.raise_for_status()
        raw = r.text
        d = json.loads(raw[raw.index("{") : raw.rindex("}") + 1])
        result = {
            "基金代码": d.get("fundcode", fund_code), "基金名称": d.get("name", "N/A"),
            "最新净值": d.get("dwjz", "N/A"), "净值日期": d.get("jzrq", "N/A"),
            "估算净值": d.get("gsz", "N/A"), "估算涨幅(%)": d.get("gszzl", "N/A"),
        }
        return json.dumps(result, ensure_ascii=False, indent=2) + "\n\n⚠️ 基金过往业绩不代表未来表现。"
    except Exception as e:
        return f"获取基金信息失败: {str(e)[:100]}"


# ═══════════════════════════════════════════════════════════════
# 工具 7：金融计算器
# ═══════════════════════════════════════════════════════════════

@tool
def calculate_financial(
    calc_type: str = "",
    principal: float = 0, rate: float = 0, years: float = 1,
    buy_price: float = 0, sell_price: float = 0,
    stock_price: float = 0, eps: float = 0, bps: float = 0,
    total_return: float = 0, days: float = 365,
    total_asset: float = 0, risk_percent: float = 2, stop_loss: float = 5,
) -> str:
    """
    执行常用金融计算。calc_type 必填：
    - compound: 复利计算（需要 principal本金, rate年利率%, years年数）
    - simple_return: 收益率（需要 buy_price, sell_price）
    - pe_ratio: 市盈率（需要 stock_price, eps）
    - pb_ratio: 市净率（需要 stock_price, bps）
    - annualized_return: 年化收益率（需要 total_return总收益率%, days天数）
    - position_size: 仓位计算（需要 total_asset, risk_percent%, stop_loss%）
    """
    logger.info(f"[工具] calculate_financial: type={calc_type}")
    try:
        if calc_type == "compound":
            result = principal * (1 + rate / 100) ** years
            return f"【复利计算】本金 {principal:,.0f} 元 | 年利率 {rate}% | {years} 年 → 最终 {result:,.2f} 元 (收益 {result-principal:,.2f} 元)"
        elif calc_type == "simple_return":
            ret = (sell_price - buy_price) / buy_price * 100
            return f"【收益率】买入 {buy_price} 元 → 卖出 {sell_price} 元 → 收益率 {ret:.2f}%"
        elif calc_type == "pe_ratio":
            return f"【市盈率】PE = {stock_price}/{eps} = {stock_price/eps:.2f} 倍" if eps else "EPS 不能为 0"
        elif calc_type == "pb_ratio":
            return f"【市净率】PB = {stock_price}/{bps} = {stock_price/bps:.2f} 倍" if bps else "每股净资产不能为 0"
        elif calc_type == "annualized_return":
            a = ((1 + total_return / 100) ** (365 / days) - 1) * 100
            return f"【年化收益率】{total_return}% 持有 {int(days)} 天 → 年化 {a:.2f}%"
        elif calc_type == "position_size":
            max_loss = total_asset * risk_percent / 100
            pos = max_loss / (stop_loss / 100)
            return f"【仓位计算】总资产 {total_asset:,.0f} 元 | 风险 {risk_percent}% | 止损 {stop_loss}% → 仓位 {pos:,.0f} 元"
        else:
            return f"不支持的计算类型: {calc_type}，支持: compound/simple_return/pe_ratio/pb_ratio/annualized_return/position_size"
    except Exception as e:
        logger.error(f"金融计算失败: {e}")
        return f"计算出错: {str(e)}"


# ═══════════════════════════════════════════════════════════════
# 工具 8：当前时间
# ═══════════════════════════════════════════════════════════════

@tool
def get_current_datetime() -> str:
    """获取当前日期时间及 A 股交易日判断。"""
    now = datetime.now()
    wd = ["周一","周二","周三","周四","周五","周六","周日"][now.weekday()]
    trading = "是交易日（仅排除周末）" if now.weekday() < 5 else "非交易日（周末）"
    return f"{now.strftime('%Y年%m月%d日 %H:%M:%S')} {wd} | A股: {trading} | 交易时间 9:30-15:00"


# ═══════════════════════════════════════════════════════════════
# 工具注册
# ═══════════════════════════════════════════════════════════════

def get_all_tools() -> list:
    return [
        search_knowledge_base, get_stock_realtime, get_stock_history,
        get_stock_financial, get_market_index, get_fund_info,
        calculate_financial, get_current_datetime,
    ]

"""
tools/finance_tools.py - 金融 Agent 工具集
包含：
  1. search_knowledge_base  - 检索金融知识库（RAG）
  2. get_stock_realtime     - 获取 A 股实时行情
  3. get_stock_history      - 获取股票历史 K 线数据
  4. get_stock_financial    - 获取股票财务指标
  5. get_market_index       - 获取主要市场指数
  6. get_fund_info          - 获取基金基本信息
  7. calculate_financial    - 金融计算器（收益率/复利/市盈率等）
  8. get_current_datetime   - 获取当前日期时间
"""

import json
from datetime import datetime, date
from typing import List

from langchain_core.tools import tool

from services.rag_service import rag_service
from utils.logger import get_logger

logger = get_logger(__name__)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具 1：金融知识库检索（RAG）
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@tool
def search_knowledge_base(query: str) -> str:
    """
    检索金融知识库，获取专业金融知识、政策解读、投资策略等信息。
    适用场景：用户询问金融概念、行业分析方法、投资策略、风险管理等知识性问题。

    Args:
        query: 检索查询，如"什么是市盈率"、"价值投资策略"

    Returns:
        知识库中相关内容的总结
    """
    logger.info(f"[工具] search_knowledge_base: {query}")
    try:
        result = rag_service.query(query)
        return result
    except Exception as e:
        logger.error(f"知识库检索失败: {e}")
        return f"知识库检索出现错误: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具 2：A 股实时行情
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@tool
def get_stock_realtime(stock_code: str) -> str:
    """
    获取 A 股股票的实时行情数据，包括最新价、涨跌幅、成交量、市值等。
    股票代码格式：沪市 6 位数字（如 600519），深市 6 位数字（如 000001）。

    Args:
        stock_code: A 股股票代码，如 "600519"（贵州茅台）、"000001"（平安银行）

    Returns:
        实时行情数据字符串
    """
    logger.info(f"[工具] get_stock_realtime: {stock_code}")
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == stock_code]

        if row.empty:
            return f"未找到股票代码 {stock_code}，请检查代码是否正确。"

        row = row.iloc[0]
        result = {
            "股票代码": stock_code,
            "股票名称": row.get("名称", "N/A"),
            "最新价(元)": row.get("最新价", "N/A"),
            "涨跌幅(%)": row.get("涨跌幅", "N/A"),
            "涨跌额(元)": row.get("涨跌额", "N/A"),
            "成交量(手)": row.get("成交量", "N/A"),
            "成交额(元)": row.get("成交额", "N/A"),
            "今开(元)": row.get("今开", "N/A"),
            "最高(元)": row.get("最高", "N/A"),
            "最低(元)": row.get("最低", "N/A"),
            "昨收(元)": row.get("昨收", "N/A"),
            "市盈率(动)": row.get("市盈率-动态", "N/A"),
            "总市值(元)": row.get("总市值", "N/A"),
            "流通市值(元)": row.get("流通市值", "N/A"),
            "数据时间": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        return json.dumps(result, ensure_ascii=False, indent=2)

    except ImportError:
        return "akshare 库未安装，请运行: pip install akshare"
    except Exception as e:
        logger.error(f"获取股票实时行情失败: {e}")
        return f"获取行情数据失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具 3：股票历史 K 线数据
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@tool
def get_stock_history(stock_code: str, period: str = "daily", limit: int = 20) -> str:
    """
    获取 A 股股票历史 K 线数据（前复权）。

    Args:
        stock_code: 股票代码，如 "600519"
        period: K 线周期，可选 "daily"（日线）、"weekly"（周线）、"monthly"（月线）
        limit: 返回最近几个交易日/周/月的数据，默认 20，最大 60

    Returns:
        近期 K 线数据表格字符串
    """
    logger.info(f"[工具] get_stock_history: {stock_code}, period={period}, limit={limit}")
    try:
        import akshare as ak

        period_map = {"daily": "daily", "weekly": "weekly", "monthly": "monthly"}
        ak_period = period_map.get(period, "daily")

        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period=ak_period,
            adjust="qfq",  # 前复权
        )

        if df.empty:
            return f"股票 {stock_code} 暂无历史数据。"

        # 取最近 N 条
        limit = min(limit, 60)
        df = df.tail(limit)[["日期", "开盘", "收盘", "最高", "最低", "成交量", "涨跌幅"]]

        # 计算简单统计
        close_prices = df["收盘"].astype(float)
        summary = {
            "股票代码": stock_code,
            "周期": period,
            "数据条数": len(df),
            "区间最高价": float(close_prices.max()),
            "区间最低价": float(close_prices.min()),
            "区间平均价": round(float(close_prices.mean()), 2),
            "最新收盘价": float(close_prices.iloc[-1]),
            "区间涨跌幅(%)": round(
                (float(close_prices.iloc[-1]) - float(close_prices.iloc[0]))
                / float(close_prices.iloc[0]) * 100, 2
            ),
        }

        output = f"【{stock_code} 历史行情统计】\n"
        output += json.dumps(summary, ensure_ascii=False, indent=2)
        output += f"\n\n【近 {limit} 个交易单位数据】\n"
        output += df.to_string(index=False)
        return output

    except ImportError:
        return "akshare 库未安装，请运行: pip install akshare"
    except Exception as e:
        logger.error(f"获取历史数据失败: {e}")
        return f"获取历史数据失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具 4：股票财务指标
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@tool
def get_stock_financial(stock_code: str) -> str:
    """
    获取 A 股股票的核心财务指标，包括 PE、PB、ROE、净利率等，用于基本面分析。

    Args:
        stock_code: 股票代码，如 "600519"

    Returns:
        财务指标摘要字符串
    """
    logger.info(f"[工具] get_stock_financial: {stock_code}")
    try:
        import akshare as ak

        df = ak.stock_zh_a_spot_em()
        row = df[df["代码"] == stock_code]

        if row.empty:
            return f"未找到股票 {stock_code} 的财务数据。"

        row = row.iloc[0]
        financial = {
            "股票代码": stock_code,
            "股票名称": row.get("名称", "N/A"),
            "市盈率(动态)": row.get("市盈率-动态", "N/A"),
            "市净率": row.get("市净率", "N/A"),
            "总市值(亿元)": (
                round(float(row.get("总市值", 0)) / 1e8, 2)
                if row.get("总市值") else "N/A"
            ),
            "流通市值(亿元)": (
                round(float(row.get("流通市值", 0)) / 1e8, 2)
                if row.get("流通市值") else "N/A"
            ),
            "换手率(%)": row.get("换手率", "N/A"),
            "量比": row.get("量比", "N/A"),
            "数据时间": datetime.now().strftime("%Y-%m-%d"),
        }

        output = json.dumps(financial, ensure_ascii=False, indent=2)
        output += "\n\n⚠️ 以上财务数据仅供参考，不构成投资建议，投资有风险，入市须谨慎。"
        return output

    except ImportError:
        return "akshare 库未安装，请运行: pip install akshare"
    except Exception as e:
        logger.error(f"获取财务指标失败: {e}")
        return f"获取财务指标失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具 5：主要市场指数行情
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@tool
def get_market_index() -> str:
    """
    获取当前主要股票市场指数的实时行情，包括上证指数、深证成指、创业板指、科创50等。
    当用户询问"大盘"、"市场行情"、"指数"时使用此工具。

    Returns:
        各主要指数的最新行情数据
    """
    logger.info("[工具] get_market_index")
    try:
        import akshare as ak

        # 要查询的指数列表
        index_codes = {
            "000001": "上证指数",
            "399001": "深证成指",
            "399006": "创业板指",
            "000688": "科创50",
            "000300": "沪深300",
            "000016": "上证50",
        }

        df = ak.stock_zh_index_spot_em()
        results = []

        for code, name in index_codes.items():
            row = df[df["代码"] == code]
            if not row.empty:
                r = row.iloc[0]
                results.append({
                    "名称": name,
                    "代码": code,
                    "最新点位": r.get("最新价", "N/A"),
                    "涨跌幅(%)": r.get("涨跌幅", "N/A"),
                    "涨跌额": r.get("涨跌额", "N/A"),
                    "成交额(亿)": (
                        round(float(r.get("成交额", 0)) / 1e8, 2)
                        if r.get("成交额") else "N/A"
                    ),
                })

        output = f"【主要指数实时行情 {datetime.now().strftime('%Y-%m-%d %H:%M')}】\n\n"
        for idx in results:
            output += (
                f"📊 {idx['名称']}（{idx['代码']}）\n"
                f"   点位: {idx['最新点位']}  涨跌幅: {idx['涨跌幅(%)']}%\n"
                f"   涨跌额: {idx['涨跌额']}  成交额: {idx['成交额(亿)']}亿\n\n"
            )
        return output

    except ImportError:
        return "akshare 库未安装，请运行: pip install akshare"
    except Exception as e:
        logger.error(f"获取指数行情失败: {e}")
        return f"获取指数行情失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具 6：基金信息查询
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@tool
def get_fund_info(fund_code: str) -> str:
    """
    获取公募基金的基本信息和近期净值数据。

    Args:
        fund_code: 基金代码（6位数字），如 "000001"（华夏成长）、"110022"（易方达消费）

    Returns:
        基金基本信息和净值数据字符串
    """
    logger.info(f"[工具] get_fund_info: {fund_code}")
    try:
        import akshare as ak

        # 获取基金净值
        df = ak.fund_open_fund_info_em(fund=fund_code, indicator="单位净值走势")

        if df.empty:
            return f"未找到基金代码 {fund_code} 的数据，请检查代码是否正确。"

        # 取最近 10 个交易日
        recent = df.tail(10)
        latest = df.iloc[-1]

        # 计算近 1 月、近 3 月涨跌幅
        def calc_return(days):
            if len(df) >= days:
                past = df.iloc[-days]["单位净值"]
                now = df.iloc[-1]["单位净值"]
                return round((float(now) - float(past)) / float(past) * 100, 2)
            return "N/A"

        summary = {
            "基金代码": fund_code,
            "最新净值": float(latest["单位净值"]),
            "净值日期": str(latest["净值日期"]),
            "近1月涨跌幅(%)": calc_return(21),
            "近3月涨跌幅(%)": calc_return(63),
            "近10日净值数据": recent[["净值日期", "单位净值", "日增长率"]].to_dict("records"),
        }

        output = json.dumps(summary, ensure_ascii=False, indent=2, default=str)
        output += "\n\n⚠️ 基金过往业绩不代表未来表现，投资有风险，入市须谨慎。"
        return output

    except ImportError:
        return "akshare 库未安装，请运行: pip install akshare"
    except Exception as e:
        logger.error(f"获取基金信息失败: {e}")
        return f"获取基金信息失败: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具 7：金融计算器
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@tool
def calculate_financial(params: str = "") -> str:
    """
    执行常用金融计算。传入一个 JSON 对象，包含 calc_type 和对应参数。

    支持的计算类型：
    - calc_type="compound"：复利计算（参数：principal本金, rate年利率%, years年数）
      示例: {"calc_type": "compound", "principal": 100000, "rate": 8, "years": 10}
    - calc_type="simple_return"：简单收益率（参数：buy_price买入价, sell_price卖出价）
      示例: {"calc_type": "simple_return", "buy_price": 50, "sell_price": 55}
    - calc_type="pe_ratio"：市盈率（参数：stock_price股价, eps每股收益）
      示例: {"calc_type": "pe_ratio", "stock_price": 100, "eps": 5}
    - calc_type="pb_ratio"：市净率（参数：stock_price股价, bps每股净资产）
      示例: {"calc_type": "pb_ratio", "stock_price": 100, "bps": 20}
    - calc_type="annualized_return"：年化收益率（参数：total_return总收益率%, days持有天数）
      示例: {"calc_type": "annualized_return", "total_return": 15, "days": 180}
    - calc_type="position_size"：仓位计算（参数：total_asset总资产, risk_percent风险比例%, stop_loss止损比例%）
      示例: {"calc_type": "position_size", "total_asset": 500000, "risk_percent": 2, "stop_loss": 5}

    Args:
        params: JSON 格式的参数字符串，如 '{"calc_type":"compound","principal":100000,"rate":8,"years":10}'

    Returns:
        计算结果字符串
    """
    import json as _json

    # ── 解析参数 ──────────────────────────────────────────
    try:
        if isinstance(params, str):
            p = _json.loads(params)
        elif isinstance(params, dict):
            p = params
        else:
            return f"参数格式错误，需要 JSON 字符串或对象，收到: {type(params).__name__}"
    except (_json.JSONDecodeError, TypeError) as e:
        return f"参数解析失败: {str(e)}，请提供有效的 JSON，如 '{{\"calc_type\":\"compound\",\"principal\":100000,\"rate\":8,\"years\":10}}'"

    calc_type = p.pop("calc_type", "")
    kwargs = p  # 剩余字段即为计算参数

    logger.info(f"[工具] calculate_financial: type={calc_type}, kwargs={kwargs}")

    try:
        if calc_type == "compound":
            principal = float(kwargs.get("principal", 0))
            rate = float(kwargs.get("rate", 0)) / 100
            years = float(kwargs.get("years", 1))
            result = principal * (1 + rate) ** years
            total_profit = result - principal
            return (
                f"【复利计算结果】\n"
                f"  本金: {principal:,.2f} 元\n"
                f"  年利率: {kwargs.get('rate')}%\n"
                f"  投资年数: {years} 年\n"
                f"  最终金额: {result:,.2f} 元\n"
                f"  总收益: {total_profit:,.2f} 元 ({total_profit/principal*100:.2f}%)"
            )

        elif calc_type == "simple_return":
            buy = float(kwargs.get("buy_price", 0))
            sell = float(kwargs.get("sell_price", 0))
            ret = (sell - buy) / buy * 100
            return (
                f"【收益率计算】\n"
                f"  买入价: {buy} 元\n"
                f"  卖出价: {sell} 元\n"
                f"  收益率: {ret:.2f}%\n"
                f"  {'盈利' if ret >= 0 else '亏损'}: {abs(sell-buy):.2f} 元/股"
            )

        elif calc_type == "pe_ratio":
            price = float(kwargs.get("stock_price", 0))
            eps = float(kwargs.get("eps", 0))
            if eps == 0:
                return "EPS 不能为 0"
            pe = price / eps
            return (
                f"【市盈率(PE)计算】\n"
                f"  股价: {price} 元\n"
                f"  每股收益(EPS): {eps} 元\n"
                f"  市盈率: {pe:.2f} 倍\n"
                f"  {'估值偏高' if pe > 30 else '估值适中' if pe > 15 else '估值偏低'}（仅作参考，需结合行业对比）"
            )

        elif calc_type == "pb_ratio":
            price = float(kwargs.get("stock_price", 0))
            bps = float(kwargs.get("bps", 0))
            if bps == 0:
                return "每股净资产不能为 0"
            pb = price / bps
            return (
                f"【市净率(PB)计算】\n"
                f"  股价: {price} 元\n"
                f"  每股净资产(BPS): {bps} 元\n"
                f"  市净率: {pb:.2f} 倍"
            )

        elif calc_type == "annualized_return":
            total_return = float(kwargs.get("total_return", 0)) / 100
            days = float(kwargs.get("days", 365))
            annualized = ((1 + total_return) ** (365 / days) - 1) * 100
            return (
                f"【年化收益率计算】\n"
                f"  持有期总收益: {kwargs.get('total_return')}%\n"
                f"  持有天数: {int(days)} 天\n"
                f"  年化收益率: {annualized:.2f}%"
            )

        elif calc_type == "position_size":
            total_asset = float(kwargs.get("total_asset", 0))
            risk_percent = float(kwargs.get("risk_percent", 2)) / 100
            stop_loss = float(kwargs.get("stop_loss", 5)) / 100
            max_loss = total_asset * risk_percent
            position = max_loss / stop_loss
            return (
                f"【仓位计算（2% 风险法则）】\n"
                f"  总资产: {total_asset:,.2f} 元\n"
                f"  单笔风险比例: {kwargs.get('risk_percent')}%\n"
                f"  止损比例: {kwargs.get('stop_loss')}%\n"
                f"  最大可接受亏损: {max_loss:,.2f} 元\n"
                f"  建议最大仓位: {position:,.2f} 元 ({position/total_asset*100:.1f}%)"
            )

        else:
            return (
                f"不支持的计算类型: {calc_type}\n"
                f"支持的类型: compound, simple_return, pe_ratio, pb_ratio, "
                f"annualized_return, position_size"
            )

    except Exception as e:
        logger.error(f"金融计算失败: {e}")
        return f"计算出错: {str(e)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 工具 8：获取当前时间
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@tool
def get_current_datetime() -> str:
    """
    获取当前的日期和时间，以及今天是否为 A 股交易日（周一至周五）。

    Returns:
        当前日期时间字符串及交易日判断
    """
    now = datetime.now()
    weekday = now.weekday()  # 0=周一 ... 6=周日
    weekday_names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    is_trading_day = weekday < 5  # 简单判断，未考虑节假日

    return (
        f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')}\n"
        f"星期: {weekday_names[weekday]}\n"
        f"A股交易日判断: {'是交易日（仅排除周末，未排除节假日）' if is_trading_day else '非交易日（周末）'}\n"
        f"A股交易时间: 9:30-11:30, 13:00-15:00"
    )


# ── 工具列表 ────────────────────────────────────────────────────

def get_all_tools() -> list:
    """返回所有可用工具列表，注册到 Agent。"""
    return [
        search_knowledge_base,
        get_stock_realtime,
        get_stock_history,
        get_stock_financial,
        get_market_index,
        get_fund_info,
        calculate_financial,
        get_current_datetime,
    ]

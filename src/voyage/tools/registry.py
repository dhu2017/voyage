"""Tool registry — central catalog of all MCP tools."""

from typing import Callable

TOOLS: dict[str, dict] = {}


def register(name: str, description: str, input_schema: dict, handler: Callable):
    """Register an MCP tool."""
    TOOLS[name] = {
        "name": name,
        "description": description,
        "inputSchema": input_schema,
        "handler": handler,
    }


def get_tool(name: str) -> dict | None:
    return TOOLS.get(name)


def list_tools() -> list[dict]:
    return [
        {"name": t["name"], "description": t["description"], "inputSchema": t["inputSchema"]}
        for t in TOOLS.values()
    ]


# ── Register all tools ──────────────────────────────────────────────

from .data import get_index_data, list_local_data, refresh_local_data
from .optimize import optimize_portfolio
from .backtest import run_backtest
from .metrics import calc_metrics, correlation_matrix

register(
    name="get_index_data",
    description="获取指数历史行情数据。支持A股/H股/美股/欧股/债券/黄金等大类资产。返回时序数据和基础统计摘要。",
    input_schema={
        "type": "object",
        "properties": {
            "index": {
                "type": "string",
                "description": "指数名称或代码，如 '沪深300', 'SPX', '黄金', 'NDX'"
            },
            "start_date": {
                "type": "string",
                "description": "起始日期，格式 YYYY-MM-DD"
            },
            "end_date": {
                "type": "string",
                "description": "结束日期，格式 YYYY-MM-DD（可选，默认今天）"
            },
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "返回字段，默认 ['close']，可选 ['open','high','low','close','volume']"
            },
            "summary_only": {
                "type": "boolean",
                "description": "仅返回摘要统计，不返回时序数据（用于缓存刷新场景）"
            },
        },
        "required": ["index", "start_date"],
    },
    handler=get_index_data,
)

register(
    name="optimize_portfolio",
    description="给定资产池，计算最优权重配比。支持 risk_parity / mvo / hrp / equal_weight 方法。",
    input_schema={
        "type": "object",
        "properties": {
            "assets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "资产列表，如 ['沪深300', '纳斯达克100', '黄金', '中国10年国债']"
            },
            "method": {
                "type": "string",
                "enum": ["risk_parity", "mvo", "hrp", "equal_weight"],
                "description": "优化方法，默认 risk_parity"
            },
            "risk_free_rate": {
                "type": "number",
                "description": "无风险利率，默认 0.02"
            },
            "lookback_days": {
                "type": "integer",
                "description": "回看窗口（交易日），默认 252"
            },
            "constraints": {
                "type": "object",
                "description": "约束条件，如 {max_weight: 0.4, min_weight: 0.05}"
            },
        },
        "required": ["assets"],
    },
    handler=optimize_portfolio,
)

register(
    name="run_backtest",
    description="对给定的资产权重组合进行历史回测。支持按月/季/年调仓，含交易成本和基准对比。",
    input_schema={
        "type": "object",
        "properties": {
            "weights": {
                "type": "object",
                "description": "资产权重，如 {'沪深300': 0.3, '纳斯达克100': 0.3, '黄金': 0.2, '中国10年国债': 0.2}"
            },
            "start_date": {
                "type": "string",
                "description": "回测起始日期 YYYY-MM-DD"
            },
            "end_date": {
                "type": "string",
                "description": "回测结束日期 YYYY-MM-DD"
            },
            "rebalance": {
                "type": "string",
                "enum": ["monthly", "quarterly", "yearly", "none"],
                "description": "调仓频率，默认 quarterly"
            },
            "initial_capital": {
                "type": "number",
                "description": "初始资金，默认 1000000"
            },
            "transaction_cost": {
                "type": "number",
                "description": "交易成本比例，默认 0.001"
            },
            "benchmark": {
                "type": "string",
                "description": "基准指数名称（可选）"
            },
        },
        "required": ["weights", "start_date", "end_date"],
    },
    handler=run_backtest,
)

register(
    name="calc_metrics",
    description="计算单个或多个资产的风险收益指标：Sharpe / Sortino / MaxDD / 年化收益 / 年化波动 / Calmar / Alpha / Beta。",
    input_schema={
        "type": "object",
        "properties": {
            "targets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "资产名称列表"
            },
            "period": {
                "type": "string",
                "enum": ["1y", "3y", "5y", "ytd", "max"],
                "description": "计算区间，默认 1y"
            },
            "benchmark": {
                "type": "string",
                "description": "基准指数名称（可选）"
            },
            "risk_free_rate": {
                "type": "number",
                "description": "无风险利率，默认 0.02"
            },
        },
        "required": ["targets"],
    },
    handler=calc_metrics,
)

register(
    name="correlation_matrix",
    description="计算资产间相关性矩阵，支持 Pearson / Spearman 方法。返回矩阵数据和高/低相关资产对提示。",
    input_schema={
        "type": "object",
        "properties": {
            "assets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "资产名称列表"
            },
            "period": {
                "type": "string",
                "enum": ["1y", "3y", "5y"],
                "description": "计算区间，默认 1y"
            },
            "method": {
                "type": "string",
                "enum": ["pearson", "spearman"],
                "description": "相关性计算方法，默认 pearson"
            },
        },
        "required": ["assets"],
    },
    handler=correlation_matrix,
)

register(
    name="list_local_data",
    description="列出本地可用的数据源快照：通达信 vipdoc 目录下的 .day 文件、项目 parquet 缓存，以及每个已登记指数对应的本地命中情况。用于快速确认哪些指数无需联网即可使用。",
    input_schema={
        "type": "object",
        "properties": {},
    },
    handler=list_local_data,
)

register(
    name="refresh_local_data",
    description="通过 mootdx 直连通达信行情服务器，把本地 TDX 指数数据补到最新。不传参时刷新所有已登记的 A股指数（沪深300/中证500/中证2000/上证50 等），传 index 参数则只刷新指定指数。结果写入 parquet 缓存，不改 vipdoc 目录。外盘/黄金/美债等远程资产会被跳过。",
    input_schema={
        "type": "object",
        "properties": {
            "index": {
                "type": "string",
                "description": "可选，指定要刷新的指数名称或别名，如 '沪深300'。不传则全部刷新。",
            },
        },
    },
    handler=refresh_local_data,
)

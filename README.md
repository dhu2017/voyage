# Voyage

个人投资组合管理工具，基于 MCP (Model Context Protocol) 协议，为 Claude Code 提供 A 股/港股/美股指数数据查询、组合优化、历史回测和风险指标计算能力。

## 功能

- **行情数据** — 获取沪深300、纳斯达克100、黄金、国债等指数历史数据，支持通达信本地数据和远程 API
- **组合优化** — 支持 Risk Parity / MVO / HRP / Equal Weight 方法计算最优权重
- **历史回测** — 对给定权重组合进行历史模拟，支持按月/季/年调仓，含交易成本和基准对比
- **风险指标** — 计算 Sharpe / Sortino / MaxDD / 年化收益 / Alpha / Beta 等指标
- **相关性分析** — 资产间 Pearson / Spearman 相关性矩阵
- **事件研究** — 扫描历史数据中满足条件的事件，分析事件后表现和胜率

## 快速开始

```bash
# 安装依赖
pip install -e .

# 运行 MCP 服务器
python -m voyage.server
```

在 Claude Code 中配置 `.mcp.json`：

```json
{
  "mcpServers": {
    "voyage-invest": {
      "command": "python",
      "args": ["-m", "voyage.server"],
      "cwd": "/path/to/voyage/src",
      "env": {
        "PYTHONPATH": "/path/to/voyage/src"
      }
    }
  }
}
```

## 项目结构

```
src/voyage/
├── server.py          # MCP 服务器入口
├── engine/            # 核心计算引擎
│   ├── backtester.py  # 回测引擎
│   ├── indicators.py  # 技术指标
│   └── portfolio.py   # 组合优化
└── tools/             # MCP 工具定义
    ├── backtest.py    # 回测工具
    ├── data.py        # 数据获取
    ├── metrics.py     # 风险指标
    └── optimize.py    # 组合优化
```

## 依赖

- Python >= 3.10
- mcp, pandas, numpy, scipy
- akshare, yfinance, mootdx (数据源)
- riskfolio-lib (组合优化)

## License

GNU General Public License v3.0

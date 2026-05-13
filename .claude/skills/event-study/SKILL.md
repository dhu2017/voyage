---
name: event-study
description: "事件驱动统计分析。扫描历史数据中满足条件的事件，分析事件后表现、胜率和统计显著性。当用户提到概率、胜率、之后表现、事件、统计验证、涨了之后、跌了之后、创新高之后、破位之后、放量之后等关键词时触发。用户说'3天涨20%后跑赢大盘的概率'、'创新高后一个月表现'、'大跌之后反弹概率'都应该使用这个 skill。"
---

# /event-study — 事件驱动统计分析

用户想知道"某种事件发生后，资产表现如何"时，按以下流程操作。

## 核心思路

事件条件是无限的（涨跌幅、均线交叉、量价背离、跨资产联动...），不可能穷举为固定参数。所以：
- **MCP 只负责数据**：用 `get_index_data` 确保缓存最新
- **Claude Code 动态生成 Python 脚本**：用 pandas 写扫描逻辑，调用 `voyage.engine.event_helpers` 做统计
- **Bash 执行脚本**，解读 stdout 输出回答用户

## 执行步骤

### 1. 确保数据缓存

先调用 `get_index_data` MCP tool 获取所需资产数据（和基准数据，如需对比），加上 `summary_only=true` 避免全量时序数据返回到上下文。这会自动更新 parquet 缓存。

- 起始日期尽量早（如 `2005-01-01`），事件分析需要足够长的历史
- 如果用户没指定基准，默认用沪深300（A股资产）或标普500（美股资产）

### 2. 生成 Python 分析脚本

写一个 Python 脚本到 `/tmp/event_analysis_xxx.py`，结构如下：

```python
import sys
import json
import pandas as pd
import numpy as np
from voyage.engine.event_helpers import (
    load_prices, post_event_returns, return_stats,
    benchmark_comparison, decluster
)

# 1. 加载数据
prices = load_prices("资产名称", start_date="2005-01-01")

# 2. 扫描事件（这部分根据用户需求动态生成）
# 例：3个交易日涨幅超过20%
rolling_ret = prices.pct_change(3)
event_mask = rolling_ret > 0.20
event_dates = prices.index[event_mask]

# 3. 去聚集（避免连续触发的事件重复计算）
event_dates = decluster(event_dates, prices.index, min_gap=5)

# 4. 计算事件后收益
results = {}
for period_name, days in [("1周", 5), ("1个月", 20), ("3个月", 60)]:
    rets = post_event_returns(prices, event_dates, holding_period=days)
    results[period_name] = return_stats(rets)

# 5. 基准对比（如需）
benchmark = load_prices("基准名称", start_date="2005-01-01")
for period_name, days in [("1周", 5), ("1个月", 20), ("3个月", 60)]:
    asset_rets = post_event_returns(prices, event_dates, holding_period=days)
    bench_rets = post_event_returns(benchmark, event_dates, holding_period=days)
    # 对齐：只保留两者都有数据的事件
    min_len = min(len(asset_rets), len(bench_rets))
    if min_len > 0:
        results[f"{period_name}_vs_benchmark"] = benchmark_comparison(
            asset_rets[:min_len], bench_rets[:min_len]
        )

# 6. 输出
output = {
    "event_count": len(event_dates),
    "first_event": str(event_dates[0].date()) if len(event_dates) > 0 else None,
    "last_event": str(event_dates[-1].date()) if len(event_dates) > 0 else None,
    "event_dates": [str(d.date()) for d in event_dates],
    "results": results,
}
print(json.dumps(output, ensure_ascii=False, indent=2))
```

### 3. 执行脚本

```bash
cd D:/Arena/voyage/src && PYTHONPATH=. python /tmp/event_analysis_xxx.py
```

### 4. 解读结果并回答用户

## 常见事件条件的 pandas 写法

供生成脚本时参考：

| 事件描述 | pandas 代码 |
|---------|------------|
| N日涨幅超过X% | `prices.pct_change(N) > X/100` |
| N日跌幅超过X% | `prices.pct_change(N) < -X/100` |
| 创N日新高 | `prices == prices.rolling(N).max()` |
| 创N日新低 | `prices == prices.rolling(N).min()` |
| 单日涨幅超过X% | `prices.pct_change(1) > X/100` |
| 单日跌幅超过X% | `prices.pct_change(1) < -X/100` |
| 从高点回撤超过X% | `prices / prices.cummax() - 1 < -X/100` |
| 均线金叉(短上穿长) | `(prices.rolling(short).mean() > prices.rolling(long).mean()) & (prices.rolling(short).mean().shift(1) <= prices.rolling(long).mean().shift(1))` |
| 连续N日上涨 | `(prices.pct_change(1) > 0).rolling(N).sum() == N` |
| 放量(成交量>N日均量X倍) | 需要 volume 数据：`volume > volume.rolling(N).mean() * X` |

## 持有期默认值

用户没明确说持有期时，默认分析多个窗口：
- 1周（5个交易日）
- 1个月（20个交易日）
- 3个月（60个交易日）

## 去聚集参数

默认 `min_gap` = 最短持有期的交易日数。例如分析1个月持有期，min_gap=20。
这避免重叠的持有期窗口导致统计不独立。

## 输出格式

### 事件概览

> 共扫描到 **N** 次事件，时间跨度 YYYY-MM-DD 至 YYYY-MM-DD。去聚集后保留 **M** 次独立事件。

### 各持有期统计表

| 持有期 | 事件数 | 平均收益 | 中位收益 | 胜率 | 最大 | 最小 | 标准差 |
|--------|--------|----------|----------|------|------|------|--------|
| 1周 | 12 | +1.2% | +0.8% | 66.7% | +5.3% | -3.1% | 2.4% |
| 1个月 | 12 | +3.5% | +2.1% | 75.0% | +12.1% | -6.2% | 5.1% |
| 3个月 | 10 | +5.8% | +4.2% | 70.0% | +18.3% | -8.5% | 7.2% |

### 基准对比（如有）

| 持有期 | 超额均值 | 跑赢概率 | t统计量 | p值 | 显著? |
|--------|----------|----------|---------|-----|-------|
| 1个月 | +2.1% | 66.7% | 1.85 | 0.09 | 否 |

### 结论

2-3 句话总结：
- 事件发生的频率和分布
- 事件后的收益特征（正/负、显著性）
- 相对基准的超额表现
- 样本量是否足够支撑结论

## 注意事项

- 脚本输出用 `json.dumps` 打印到 stdout，方便解析
- 事件数太少（<5）时，提醒用户统计意义有限
- 如果 `load_prices` 报错说缓存不存在，说明第1步的 MCP 调用没成功，需要重新获取数据
- 去聚集很重要 — 不去聚集的话，连续触发的事件会导致持有期高度重叠，统计检验失效
- 所有百分比在输出时乘以100显示（如 0.03 → 3.0%）

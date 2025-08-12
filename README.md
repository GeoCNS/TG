# A股短线策略回测

## 快速开始

- 安装依赖：
```bash
pip install -r requirements.txt
```

- 运行回测（示例：CSI300，2019-01-01 至 2024-12-31）：
```bash
python run_backtest.py --start 2019-01-01 --end 2024-12-31 --universe csi300 --max_positions 5 --hold_days 3
```

- 结果输出：
  - `output/equity_curve.png`: 权益曲线
  - `output/trades.csv`: 交易明细
  - 控制台打印：核心指标（年化、夏普、最大回撤、胜率等）

## 策略描述（默认）
- 标的池：CSI300（可扩展）
- 入场：
  - 收盘价突破过去20日最高价（带0.5%缓冲）
  - 当日成交量 ≥ 过去20日均量的2倍
  - 当日涨幅在 [3%, 7%] 之间
  - 排除 ST/退市类股票、上市小于60个交易日的新股
- 排序：按量能放大倍数（volume_ratio）由高到低择优
- 仓位：等权，最多 `--max_positions` 只
- 交易：T+1，在信号日的下一个交易日开盘买入/卖出
- 离场：
  - 止盈：+8%
  - 止损：-4%
  - 时间止盈：持有 `--hold_days` 天
  - 采用收盘判断，次日开盘执行
- 费用与滑点（可调）：
  - 手续费 0.03%（双边），印花税 0.1%（仅卖出），滑点 0.05%

## 重要说明
- 使用 AkShare 拉取数据，默认前复权日线。
- 当前示例使用当前成分的 CSI300 存在存活偏差，仅用于演示；若需严格历史成分，请接入带历史成分数据的源。
- 回测撮合采用“收盘判定、次日开盘成交”的日频近似，未做盘中止盈止损撮合。

## 目录结构
```
workspace/
  requirements.txt
  README.md
  run_backtest.py
  src/
    config.py
    data_loader.py
    strategy_breakout.py
    backtester.py
  data/               # 本地缓存
  output/             # 输出图表和交易
```
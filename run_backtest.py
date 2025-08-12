import argparse
import os
import pandas as pd
from src.config import StrategyConfig, TradingCostConfig, RunConfig
from src.data_loader import load_universe_data, get_trading_calendar
from src.strategy_breakout import make_signal_panels
from src.backtester import DailyOpenNextBacktester


def parse_args():
    parser = argparse.ArgumentParser(description="A股短线突破策略回测")
    parser.add_argument("--start", type=str, default="2019-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--universe", type=str, default="csi300", choices=["csi300", "all"])
    parser.add_argument("--max_positions", type=int, default=5)
    parser.add_argument("--hold_days", type=int, default=3)
    parser.add_argument("--take_profit", type=float, default=0.08)
    parser.add_argument("--stop_loss", type=float, default=0.04)
    parser.add_argument("--min_vr", type=float, default=2.0, help="最小量比（20日均量倍数）")
    parser.add_argument("--limit", type=int, default=30, help="限制股票数量用于快速烟雾测试")
    parser.add_argument("--out", type=str, default="./output")
    parser.add_argument("--cash", type=float, default=1_000_000.0)
    return parser.parse_args()


def main():
    args = parse_args()

    strat_cfg = StrategyConfig(
        max_positions=args.max_positions,
        hold_days=args.hold_days,
        take_profit=args.take_profit,
        stop_loss=args.stop_loss,
        min_volume_ratio=args.min_vr,
    )
    run_cfg = RunConfig(start_date=args.start, end_date=args.end, output_dir=args.out)
    cost_cfg = TradingCostConfig()

    os.makedirs(run_cfg.output_dir, exist_ok=True)

    print("Loading trading calendar...")
    cal = get_trading_calendar(run_cfg.start_date, run_cfg.end_date)

    print(f"Loading universe data: {args.universe}")
    data = load_universe_data(args.universe, run_cfg.start_date, run_cfg.end_date, limit=args.limit)
    if not data:
        raise SystemExit("Empty universe data. Please check network or AkShare setup.")

    print("Computing signals...")
    panels = make_signal_panels(data, strat_cfg)

    print("Running backtest...")
    bt = DailyOpenNextBacktester(
        signal_panels=panels,
        trading_calendar=cal,
        strategy_cfg=strat_cfg,
        cost_cfg=cost_cfg,
        initial_cash=args.cash,
        output_dir=run_cfg.output_dir,
    )
    bt.run()
    bt.save_outputs()


if __name__ == "__main__":
    main()
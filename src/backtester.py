from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import os
import math
from collections import defaultdict
import matplotlib.pyplot as plt

from src.config import StrategyConfig, TradingCostConfig, RunConfig


@dataclass
class Position:
    code: str
    entry_date: pd.Timestamp
    entry_price: float
    shares: int
    holding_days: int = 0
    max_price_since_entry: float = 0.0


class DailyOpenNextBacktester:
    def __init__(
        self,
        signal_panels: Dict[str, pd.DataFrame],
        trading_calendar: pd.DatetimeIndex,
        strategy_cfg: StrategyConfig,
        cost_cfg: TradingCostConfig,
        initial_cash: float = 1_000_000.0,
        output_dir: str = "./output",
    ) -> None:
        self.signal_panels = signal_panels
        self.calendar = pd.DatetimeIndex(trading_calendar)
        self.strategy_cfg = strategy_cfg
        self.cost_cfg = cost_cfg
        self.initial_cash = initial_cash
        self.output_dir = output_dir

        self._prepare_price_frames()

        self.cash: float = initial_cash
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[Tuple[pd.Timestamp, float]] = []
        self.trades: List[Dict] = []

    def _prepare_price_frames(self) -> None:
        # 对齐各标的到统一交易日历，主要用于取开盘/收盘价
        aligned: Dict[str, pd.DataFrame] = {}
        for code, df in self.signal_panels.items():
            tmp = df.set_index("date").reindex(self.calendar)
            tmp["code"] = code
            aligned[code] = tmp
        self.aligned = aligned

    def _get_price(self, code: str, date: pd.Timestamp, field: str) -> Optional[float]:
        df = self.aligned.get(code)
        if df is None:
            return None
        val = df.loc[date, field]
        if pd.isna(val):
            return None
        return float(val)

    def _apply_cost(self, gross_cash: float, is_sell: bool) -> float:
        # 成交价已含滑点调整，这里仅处理佣金与印花税
        fee = abs(gross_cash) * self.cost_cfg.commission_rate
        tax = abs(gross_cash) * self.cost_cfg.stamp_duty_rate if is_sell else 0.0
        return gross_cash - fee - tax

    def _price_with_slippage(self, price: float, is_buy: bool) -> float:
        if math.isnan(price):
            return price
        adj = price * (1.0 + self.cost_cfg.slippage_rate * (1.0 if is_buy else -1.0))
        return adj

    def run(self) -> None:
        # 主循环：收盘判定，次日开盘成交
        dates = list(self.calendar)
        for i in range(len(dates)):
            current_date = dates[i]
            next_date = dates[i + 1] if i + 1 < len(dates) else None

            # 1) 计算当日市值，记录权益（用收盘价估值）
            equity = self.cash
            for code, pos in list(self.positions.items()):
                close_price = self._get_price(code, current_date, "close")
                if close_price is None:
                    continue
                equity += pos.shares * close_price
            self.equity_curve.append((current_date, equity))

            # 2) 收盘判定离场（在 next_date 开盘执行）
            exit_list: List[str] = []
            if next_date is not None:
                for code, pos in list(self.positions.items()):
                    close_price = self._get_price(code, current_date, "close")
                    if close_price is None or pos.entry_price <= 0:
                        continue
                    pos.holding_days += 1
                    pos.max_price_since_entry = max(pos.max_price_since_entry, close_price)

                    ret_from_entry = (close_price / pos.entry_price) - 1.0
                    # 时间止盈/止损
                    hit_take_profit = ret_from_entry >= self.strategy_cfg.take_profit
                    hit_stop_loss = ret_from_entry <= -self.strategy_cfg.stop_loss
                    hit_time_exit = pos.holding_days >= self.strategy_cfg.hold_days

                    if hit_take_profit or hit_stop_loss or hit_time_exit:
                        exit_list.append(code)

            # 3) 收盘选股入场（在 next_date 开盘执行）
            entry_candidates: List[Tuple[str, float]] = []  # (code, rank_score)
            if next_date is not None:
                for code, df in self.aligned.items():
                    if code in self.positions:
                        continue
                    row = df.loc[current_date]
                    if pd.isna(row.get("entry_signal", np.nan)) or not bool(row.get("entry_signal", False)):
                        continue
                    rank_score = row.get("rank_score", np.nan)
                    if pd.isna(rank_score):
                        continue
                    entry_candidates.append((code, float(rank_score)))

                # 排序并截断到剩余仓位
                remaining_slots = max(0, self.strategy_cfg.max_positions - len(self.positions))
                entry_candidates.sort(key=lambda x: x[1], reverse=True)
                entry_list = [c for c, _ in entry_candidates[:remaining_slots]]
            else:
                entry_list = []

            # 4) 在 next_date 开盘执行离场与入场
            if next_date is not None:
                # 先卖后买
                for code in exit_list:
                    if code not in self.positions:
                        continue
                    open_price = self._get_price(code, next_date, "open")
                    if open_price is None:
                        continue
                    exec_price = self._price_with_slippage(open_price, is_buy=False)

                    pos = self.positions.pop(code)
                    gross = exec_price * pos.shares
                    net_cash = self._apply_cost(gross_cash=gross, is_sell=True)
                    self.cash += net_cash

                    pct_ret = (exec_price / pos.entry_price) - 1.0
                    self.trades.append({
                        "date": next_date,
                        "code": code,
                        "side": "SELL",
                        "price": exec_price,
                        "shares": pos.shares,
                        "pnl": net_cash - (pos.shares * pos.entry_price),
                        "pct": pct_ret,
                    })

                # 买入
                if len(entry_list) > 0:
                    # 等权分配可用资金
                    alloc_per_pos = self.cash / max(1, len(entry_list))
                    for code in entry_list:
                        open_price = self._get_price(code, next_date, "open")
                        if open_price is None or open_price <= 0:
                            continue
                        exec_price = self._price_with_slippage(open_price, is_buy=True)

                        # 涨停无法买入的近似处理：若当日开盘涨停（用涨跌幅近似），则跳过
                        prev_close = self._get_price(code, current_date, "close")
                        if prev_close is not None and prev_close > 0 and (exec_price / prev_close - 1.0) >= 0.098:
                            continue

                        shares = int(alloc_per_pos // (exec_price * 100)) * 100  # A股手数：100股
                        if shares <= 0:
                            continue

                        gross = -exec_price * shares
                        net_cash = self._apply_cost(gross_cash=gross, is_sell=False)
                        if self.cash + net_cash < -1e-6:
                            continue
                        self.cash += net_cash

                        self.positions[code] = Position(
                            code=code,
                            entry_date=next_date,
                            entry_price=exec_price,
                            shares=shares,
                            holding_days=0,
                            max_price_since_entry=exec_price,
                        )
                        self.trades.append({
                            "date": next_date,
                            "code": code,
                            "side": "BUY",
                            "price": exec_price,
                            "shares": shares,
                            "pnl": 0.0,
                            "pct": 0.0,
                        })

        # 收尾：最后一日权益记录已在循环内完成

    def _metrics(self) -> dict:
        eq = pd.DataFrame(self.equity_curve, columns=["date", "equity"]).set_index("date").sort_index()
        daily_ret = eq["equity"].pct_change().fillna(0.0)
        # 年化（按 252 日）
        total_ret = eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1.0 if len(eq) > 0 else 0.0
        ann_ret = (1.0 + total_ret) ** (252.0 / max(1, len(eq))) - 1.0 if len(eq) > 0 else 0.0
        ann_vol = daily_ret.std() * math.sqrt(252.0) if len(eq) > 1 else 0.0
        sharpe = (ann_ret / ann_vol) if ann_vol > 1e-12 else 0.0

        # 最大回撤
        cummax = eq["equity"].cummax()
        dd = eq["equity"] / cummax - 1.0
        max_dd = dd.min() if len(dd) else 0.0

        # 交易统计
        trades_df = pd.DataFrame(self.trades)
        if trades_df.empty or "side" not in trades_df.columns:
            sell_trades = pd.DataFrame(columns=["pnl"])
            win_rate = 0.0
            avg_trade = 0.0
            num_trades = 0
        else:
            sell_trades = trades_df[trades_df["side"] == "SELL"].copy()
            win_rate = (sell_trades["pnl"] > 0).mean() if len(sell_trades) else 0.0
            avg_trade = sell_trades["pnl"].mean() if len(sell_trades) else 0.0
            num_trades = int((trades_df["side"] == "SELL").sum())

        return {
            "final_equity": float(eq["equity"].iloc[-1]) if len(eq) else self.initial_cash,
            "total_return": total_ret,
            "annual_return": ann_ret,
            "annual_vol": ann_vol,
            "sharpe": sharpe,
            "max_drawdown": float(max_dd),
            "num_trades": num_trades,
            "win_rate": float(win_rate),
            "avg_trade_pnl": float(avg_trade),
        }

    def save_outputs(self) -> dict:
        os.makedirs(self.output_dir, exist_ok=True)
        eq = pd.DataFrame(self.equity_curve, columns=["date", "equity"]).set_index("date").sort_index()
        metrics = self._metrics()

        # 图
        plt.figure(figsize=(10, 5))
        plt.plot(eq.index, eq["equity"], label="Equity")
        plt.title("Equity Curve")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        fig_path = os.path.join(self.output_dir, "equity_curve.png")
        plt.savefig(fig_path)
        plt.close()

        # 交易
        trades_df = pd.DataFrame(self.trades)
        trades_path = os.path.join(self.output_dir, "trades.csv")
        trades_df.to_csv(trades_path, index=False)

        # 指标打印
        print("\nCore metrics:")
        for k, v in metrics.items():
            if isinstance(v, float):
                if "return" in k or "drawdown" in k or k in ["win_rate", "annual_vol", "sharpe"]:
                    print(f"- {k}: {v:.4f}")
                else:
                    print(f"- {k}: {v:.2f}")
            else:
                print(f"- {k}: {v}")
        print(f"\nSaved: {fig_path}\nSaved: {trades_path}")
        return metrics
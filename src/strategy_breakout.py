from typing import Dict, Tuple
import pandas as pd
import numpy as np

from src.config import StrategyConfig


def compute_signals_per_symbol(df: pd.DataFrame, cfg: StrategyConfig) -> pd.DataFrame:
    data = df.copy()
    data = data.sort_values("date").reset_index(drop=True)

    # 基本字段
    for col in ["open", "high", "low", "close", "volume"]:
        if col not in data.columns:
            raise ValueError(f"Missing column {col}")

    # 涨跌幅（若无则计算）
    if "pct_chg" not in data.columns:
        data["pct_chg"] = data["close"].pct_change().fillna(0.0)

    # 换手率可选
    if "turnover" not in data.columns:
        data["turnover"] = np.nan

    # N日最高 + 缓冲
    highest_n = data["high"].rolling(cfg.breakout_lookback, min_periods=cfg.breakout_lookback).max().shift(1)
    breakout_threshold = highest_n * (1.0 + cfg.breakout_buffer)
    data["is_breakout"] = data["close"] > breakout_threshold

    # 量能放大
    avg_vol = data["volume"].rolling(20, min_periods=20).mean()
    data["volume_ratio"] = data["volume"] / avg_vol

    # 涨幅区间过滤
    data["within_return_band"] = (data["pct_chg"] >= cfg.min_daily_return) & (data["pct_chg"] <= cfg.max_daily_return)

    # 换手区间（若无换手，用True）
    if data["turnover"].isna().all():
        data["within_turnover_band"] = True
    else:
        data["within_turnover_band"] = (data["turnover"] >= cfg.min_turnover_rate) & (data["turnover"] <= cfg.max_turnover_rate)

    # 新股上市天数
    data["days_since_ipo"] = np.arange(len(data))
    data["enough_age"] = data["days_since_ipo"] >= cfg.min_days_since_ipo

    # 综合信号
    data["entry_signal"] = (
        data["is_breakout"].fillna(False)
        & (data["volume_ratio"] >= cfg.min_volume_ratio).fillna(False)
        & data["within_return_band"].fillna(False)
        & data["within_turnover_band"].fillna(True)
        & data["enough_age"].fillna(True)
    )

    # 排序强度：volume_ratio 优先
    data["rank_score"] = data["volume_ratio"].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return data


def make_signal_panels(universe_data: Dict[str, pd.DataFrame], cfg: StrategyConfig) -> Dict[str, pd.DataFrame]:
    panels: Dict[str, pd.DataFrame] = {}
    for code, df in universe_data.items():
        try:
            panels[code] = compute_signals_per_symbol(df, cfg)
        except Exception:
            continue
    return panels
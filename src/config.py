from dataclasses import dataclass


@dataclass
class TradingCostConfig:
    commission_rate: float = 0.0003  # 手续费（双边）
    stamp_duty_rate: float = 0.001   # 印花税（仅卖出）
    slippage_rate: float = 0.0005    # 滑点（按价格比例）


@dataclass
class StrategyConfig:
    breakout_lookback: int = 20
    breakout_buffer: float = 0.005  # 0.5%
    min_daily_return: float = 0.03
    max_daily_return: float = 0.07
    min_turnover_rate: float = 0.01
    max_turnover_rate: float = 0.20
    min_days_since_ipo: int = 60
    min_volume_ratio: float = 2.0
    max_positions: int = 5
    hold_days: int = 3
    take_profit: float = 0.08
    stop_loss: float = 0.04


@dataclass
class RunConfig:
    start_date: str = "2019-01-01"
    end_date: str = "2024-12-31"
    universe: str = "csi300"  # csi300 or all
    data_cache_dir: str = "./data"
    output_dir: str = "./output"
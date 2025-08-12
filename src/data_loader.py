import os
import time
from typing import List, Dict
import pandas as pd
import numpy as np
from tqdm import tqdm

try:
    import akshare as ak
except Exception as exc:
    ak = None


A_STOCK_LIMIT_UP_APPROX = 0.098  # 非ST涨停阈值近似


def ensure_dir_exists(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_trading_calendar(start_date: str, end_date: str) -> pd.DatetimeIndex:
    if ak is None:
        raise RuntimeError("AkShare not available. Please install akshare.")
    cal = ak.tool_trade_date_hist_sina()
    cal = pd.to_datetime(cal["trade_date"])
    mask = (cal >= pd.to_datetime(start_date)) & (cal <= pd.to_datetime(end_date))
    return pd.DatetimeIndex(cal[mask])


def get_csi300_universe() -> List[str]:
    if ak is None:
        raise RuntimeError("AkShare not available. Please install akshare.")
    df = ak.index_stock_cons(symbol="000300")
    # 兼容不同版本字段
    possible_cols = [
        "品种代码", "指数代码", "成分券代码", "代码", "股票代码"
    ]
    code_col = None
    for col in possible_cols:
        if col in df.columns:
            code_col = col
            break
    if code_col is None:
        raise RuntimeError(f"Unexpected columns for CSI300 constituents: {df.columns.tolist()}")
    codes = df[code_col].astype(str).str.zfill(6).unique().tolist()
    return sorted(codes)


def sanitize_stock_df(raw: pd.DataFrame) -> pd.DataFrame:
    # 标准化列
    rename_map = {
        "日期": "date",
        "开盘": "open",
        "最高": "high",
        "最低": "low",
        "收盘": "close",
        "成交量": "volume",
        "成交额": "amount",
        "涨跌幅": "pct_chg",
        "换手率": "turnover",
        "股票代码": "code",
        "名称": "name",
    }
    for k, v in list(rename_map.items()):
        if k not in raw.columns:
            # 某些版本字段英文
            if k == "涨跌幅" and "change_rate" in raw.columns:
                rename_map[k] = "change_rate"
            if k == "换手率" and "turnover_rate" in raw.columns:
                rename_map[k] = "turnover_rate"
    df = raw.rename(columns=rename_map).copy()
    # 日期索引
    if "date" not in df.columns:
        # 尝试通用日期字段
        for alt in ["日期", "trade_date", "date"]:
            if alt in df.columns:
                df["date"] = df[alt]
                break
    df["date"] = pd.to_datetime(df["date"])  # type: ignore
    df = df.sort_values("date").reset_index(drop=True)

    # 类型与缺失处理
    for col in ["open", "high", "low", "close", "volume", "amount"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "pct_chg" in df.columns:
        df["pct_chg"] = pd.to_numeric(df["pct_chg"], errors="coerce") / 100.0 if df["pct_chg"].abs().max() > 1.0 else pd.to_numeric(df["pct_chg"], errors="coerce")
    if "turnover" in df.columns:
        df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") / 100.0 if df["turnover"].abs().max() > 1.0 else pd.to_numeric(df["turnover"], errors="coerce")

    # 丢弃异常
    df = df.dropna(subset=["open", "high", "low", "close"]).copy()

    # 过滤 ST / 退市
    if "name" in df.columns:
        mask = ~df["name"].astype(str).str.contains(r"ST|\*ST|退")
        df = df.loc[mask].copy()

    return df


def fetch_stock_daily(code: str, start_date: str, end_date: str, adjust: str = "qfq") -> pd.DataFrame:
    if ak is None:
        raise RuntimeError("AkShare not available. Please install akshare.")

    cache_dir = os.path.join("data", "hist", adjust)
    ensure_dir_exists(cache_dir)
    cache_path = os.path.join(cache_dir, f"{code}.csv")

    df = None
    if os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path)
        except Exception:
            df = None

    if df is None or df.empty:
        # 拉取
        for _ in range(3):
            try:
                raw = ak.stock_zh_a_hist(symbol=code, period="daily", adjust=adjust, start_date=start_date.replace("-", ""), end_date=end_date.replace("-", ""))
                df = sanitize_stock_df(raw)
                df.to_csv(cache_path, index=False)
                break
            except Exception:
                time.sleep(1.5)
        if df is None:
            raise RuntimeError(f"Failed to fetch data for {code}")
    else:
        df = sanitize_stock_df(df)

    # 截取时间窗
    mask = (df["date"] >= pd.to_datetime(start_date)) & (df["date"] <= pd.to_datetime(end_date))
    return df.loc[mask].reset_index(drop=True)


def load_universe_data(universe: str, start_date: str, end_date: str, limit: int | None = None) -> Dict[str, pd.DataFrame]:
    if universe == "csi300":
        codes = get_csi300_universe()
    else:
        # 退而求其次：取沪深主板/创业板/科创板全A（可能较慢）
        if ak is None:
            raise RuntimeError("AkShare not available. Please install akshare.")
        spot = ak.stock_zh_a_spot_em()
        code_col = "代码" if "代码" in spot.columns else ("股票代码" if "股票代码" in spot.columns else None)
        if code_col is None:
            raise RuntimeError("Cannot find code column in A-share spot table.")
        codes = spot[code_col].astype(str).str.zfill(6).unique().tolist()

    if limit is not None and limit > 0:
        codes = codes[:limit]

    data: Dict[str, pd.DataFrame] = {}
    for code in tqdm(codes, desc=f"Downloading {universe} data"):
        try:
            df = fetch_stock_daily(code, start_date, end_date, adjust="qfq")
            # 新股天数过滤在策略阶段处理；此处保留全部供后续计算
            if len(df) == 0:
                continue
            data[code] = df
        except Exception:
            continue
    return data
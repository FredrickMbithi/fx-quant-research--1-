"""
src/features/volatility.py
──────────────────────────
Volatility features used across strategies.
Each function accepts a DataFrame with OHLCV columns and returns a Series.
"""
from __future__ import annotations

import pandas as pd
import numpy as np


def atr(df: pd.DataFrame, length: int = 20) -> pd.Series:
    """
    Average True Range (simple moving average of TR).
    True Range = max(high-low, |high-prev_close|, |low-prev_close|)
    """
    high  = df["high"]
    low   = df["low"]
    close = df["close"]

    prev_close = close.shift(1)
    tr = pd.concat(
        [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)

    return tr.rolling(length, min_periods=length).mean().rename(f"atr_{length}")

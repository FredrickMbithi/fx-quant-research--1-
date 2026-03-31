"""
src/data/loader.py
──────────────────
Load raw XAUUSD CSV → clean, normalised DataFrame with DatetimeIndex.
Saves processed file to data/processed/ for fast re-use.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_raw(path: str | Path) -> pd.DataFrame:
    """
    Read the raw MetaTrader-style CSV.
    Expected layout (no header):
        date | time | open | high | low | close | volume
    Returns a DataFrame with a UTC DatetimeIndex and columns:
        open, high, low, close, volume  (all float/int)
    """
    path = Path(path)
    log.info("Loading raw data from %s", path)

    df = pd.read_csv(
        path,
        header=0,           # first row is used as header (date of first bar)
        names=["date", "time", "open", "high", "low", "close", "volume"],
        dtype={
            "date":   str,
            "time":   str,
            "open":   float,
            "high":   float,
            "low":    float,
            "close":  float,
            "volume": float,
        },
    )

    # Build datetime index
    df["datetime"] = pd.to_datetime(df["date"] + " " + df["time"], format="%Y.%m.%d %H:%M")
    df = df.set_index("datetime").drop(columns=["date", "time"])
    df.index = df.index.tz_localize("UTC")
    df = df.sort_index()

    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean pipeline:
      1. Remove duplicate timestamps (keep last).
      2. Drop rows with any NaN in OHLCV.
      3. Enforce OHLC sanity (high >= max(open,close), low <= min(open,close)).
      4. Reindex to expected 15-min grid, forward-fill gaps up to 4 bars.
    """
    before = len(df)

    # 1. Duplicates
    df = df[~df.index.duplicated(keep="last")]

    # 2. NaN
    df = df.dropna(subset=["open", "high", "low", "close"])

    # 3. OHLC sanity — clamp rather than drop so we don't lose bars
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"]  = df[["low",  "open", "close"]].min(axis=1)

    # 4. Reindex to uniform 15-min grid
    full_range = pd.date_range(df.index[0], df.index[-1], freq="15min", tz="UTC")
    df = df.reindex(full_range)
    # Forward-fill OHLCV for short gaps (≤ 4 bars = 1 hour); leaves longer gaps as NaN
    df = df.ffill(limit=4)
    df = df.dropna(subset=["close"])

    after = len(df)
    log.info("Clean: %d → %d rows (removed %d)", before, after, before - after)
    return df


def load_processed(processed_path: str | Path, raw_path: str | Path) -> pd.DataFrame:
    """
    Return processed data, rebuilding from raw if the parquet cache is stale/missing.
    """
    processed_path = Path(processed_path)
    raw_path = Path(raw_path)

    if processed_path.exists():
        raw_mtime  = raw_path.stat().st_mtime
        proc_mtime = processed_path.stat().st_mtime
        if proc_mtime >= raw_mtime:
            log.info("Loading cached processed data from %s", processed_path)
            return pd.read_parquet(processed_path)

    log.info("Processing raw data …")
    df = load_raw(raw_path)
    df = clean(df)

    processed_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(processed_path)
    log.info("Saved processed data → %s", processed_path)
    return df

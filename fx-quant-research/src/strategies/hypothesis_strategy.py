"""
src/strategies/hypothesis_strategy.py
──────────────────────────────────────
Python translation of StrategyATrend_AlmaSlope (C# / cAlgo).

HYPOTHESIS
----------
An ALMA moving average's ATR-normalised first-difference (slope) contains
directional information on XAUUSD 15-minute bars.  When the slope has been
consistently positive/negative for `require_consecutive` bars, enter long/short.

OUTPUT CONTRACT
---------------
generate_signals(df, cfg) → pd.Series of {+1, -1, 0}
    +1  = enter long
    -1  = enter short
     0  = flat / no action

This module is intentionally pure:
  • No execution, no sizing, no position management.
  • Consumes only engineered features.
  • Fully testable in isolation.
"""
from __future__ import annotations

import pandas as pd

from src.features.trend import alma, alma_slope_normalised, consecutive_bars
from src.features.volatility import atr
from src.features.microstructure import session_mask


def build_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Compute all features required by the hypothesis.

    Parameters
    ----------
    df  : Clean OHLCV DataFrame with UTC DatetimeIndex.
    cfg : Full config dict (only 'strategy' and 'session_filters' keys used).

    Returns
    -------
    DataFrame with additional columns:
        atr, alma, alma_slope_norm,
        consec_up, consec_down,
        in_session
    """
    sc = cfg["strategy"]

    feat = df.copy()
    feat["atr"]            = atr(df, length=sc["atr_length"])
    feat["alma"]           = alma(df["close"], length=sc["alma_length"],
                                  offset=sc["alma_offset"], sigma=sc["alma_sigma"])
    feat["alma_slope_norm"] = alma_slope_normalised(feat["alma"], feat["atr"])

    up_cond   = feat["alma_slope_norm"] >  sc["min_slope_atr"]
    down_cond = feat["alma_slope_norm"] < -sc["min_slope_atr"]

    feat["consec_up"]   = consecutive_bars(up_cond)
    feat["consec_down"] = consecutive_bars(down_cond)
    feat["in_session"]  = session_mask(df.index, cfg.get("session_filters", {"enabled": False}))

    return feat


def generate_signals(df: pd.DataFrame, cfg: dict) -> pd.Series:
    """
    Core signal generator.

    Rules (mirroring C# OnBar logic):
      1. Compute ATR-normalised ALMA slope.
      2. Require `require_consecutive` consecutive bars above/below threshold.
      3. Only signal during active trading sessions (if session filter enabled).
      4. Allow shorts controlled by `allow_shorts` flag.

    Signals are generated on the *closed* bar and apply to the *next* bar
    (i.e., no look-ahead).

    Returns
    -------
    pd.Series[int] with values {+1, -1, 0}, same index as df.
    """
    sc   = cfg["strategy"]
    req  = sc["require_consecutive"]
    feat = build_features(df, cfg)

    signal = pd.Series(0, index=feat.index, name="signal")

    long_signal  = (feat["consec_up"]   >= req) & feat["in_session"]
    short_signal = (feat["consec_down"] >= req) & feat["in_session"] & sc["allow_shorts"]

    signal[long_signal]  =  1
    signal[short_signal] = -1

    # Shift by 1: signal on bar t is acted upon at bar t+1 open
    return signal.shift(1).fillna(0).astype(int)

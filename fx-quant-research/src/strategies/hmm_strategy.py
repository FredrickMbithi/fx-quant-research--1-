"""
src/strategies/hmm_strategy.py
────────────────────────────────
ALMA Slope strategy gated by HMM regime filter.

Signal logic (extends hypothesis_strategy.py):
  • Compute ALMA normalised slope as before.
  • Classify each bar into TRENDING (1) or RANGING (0) via HMM.
  • Only enter LONG/SHORT when regime == TRENDING.
  • When regime == RANGING: flat (signal = 0).

This isolates the regime gate — all other logic is unchanged.
"""
from __future__ import annotations

import logging

import pandas as pd

from src.features.trend import alma, alma_slope_normalised, consecutive_bars
from src.features.volatility import atr
from src.features.microstructure import session_mask
from src.features.regime import HMMRegimeFilter, TRENDING

log = logging.getLogger(__name__)


def build_features_hmm(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Same as hypothesis_strategy.build_features() but adds 'regime' column.
    """
    sc = cfg["strategy"]

    feat = df.copy()
    feat["atr"]             = atr(df, length=sc["atr_length"])
    feat["alma"]            = alma(df["close"], length=sc["alma_length"],
                                   offset=sc["alma_offset"], sigma=sc["alma_sigma"])
    feat["alma_slope_norm"] = alma_slope_normalised(feat["alma"], feat["atr"])

    up_cond   = feat["alma_slope_norm"] >  sc["min_slope_atr"]
    down_cond = feat["alma_slope_norm"] < -sc["min_slope_atr"]

    feat["consec_up"]   = consecutive_bars(up_cond)
    feat["consec_down"] = consecutive_bars(down_cond)
    feat["in_session"]  = session_mask(df.index, cfg.get("session_filters", {"enabled": False}))

    # ── HMM regime ──────────────────────────────────────────────────────
    hmm_cfg = cfg.get("hmm", {})
    hmm = HMMRegimeFilter(
        n_states    = hmm_cfg.get("n_states",    2),
        lookback    = hmm_cfg.get("lookback",    2000),
        refit_every = hmm_cfg.get("refit_every", 500),
        n_iter      = hmm_cfg.get("n_iter",      100),
    )
    log.info("Fitting HMM regime filter …")
    feat["regime"] = hmm.fit_predict(df)

    return feat


def generate_signals_hmm(
    df: pd.DataFrame,
    cfg: dict,
    features: pd.DataFrame | None = None,
) -> pd.Series:
    """
    Generate signals identical to the base strategy BUT only when
    HMM says the market is in a TRENDING regime.

    Parameters
    ----------
    features : Pre-built feature DataFrame (from build_features_hmm).
               If None, it will be computed internally — but passing it
               avoids a redundant HMM fit when you need both features and signals.

    Returns
    -------
    pd.Series[int]: {+1, -1, 0}, shifted +1 bar (no look-ahead).
    """
    sc   = cfg["strategy"]
    req  = sc["require_consecutive"]
    feat = features if features is not None else build_features_hmm(df, cfg)

    signal = pd.Series(0, index=feat.index, name="signal_hmm")

    trending     = feat["regime"] == TRENDING
    in_session   = feat["in_session"]

    long_signal  = (feat["consec_up"]   >= req) & in_session & trending
    short_signal = (feat["consec_down"] >= req) & in_session & trending & sc["allow_shorts"]

    signal[long_signal]  =  1
    signal[short_signal] = -1

    return signal.shift(1).fillna(0).astype(int)

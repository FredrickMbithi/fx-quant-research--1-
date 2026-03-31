"""
src/features/trend.py
─────────────────────
Trend features: ALMA (Arnaud Legoux Moving Average) and derived slope signals.
Exact port of the C# ComputeAlma() used in StrategyATrend_AlmaSlope.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# ALMA
# ---------------------------------------------------------------------------

def _alma_weights(length: int, offset: float, sigma: float) -> np.ndarray:
    """Pre-compute the Gaussian weight vector for a given ALMA parameterisation."""
    m = offset * (length - 1)
    s = length / sigma
    idxs = np.arange(length, dtype=float)
    w = np.exp(-((idxs - m) ** 2) / (2 * s * s))
    return w / w.sum()


def alma(
    series: pd.Series,
    length: int = 50,
    offset: float = 0.85,
    sigma: float = 6.0,
) -> pd.Series:
    """
    Arnaud Legoux Moving Average.

    Parameters
    ----------
    series  : Close prices (or any price series).
    length  : Window length (bars).
    offset  : Gaussian centre offset in [0, 1].  Higher = more recent weighting.
    sigma   : Gaussian width.  Higher = smoother (approaches SMA).

    Returns
    -------
    pd.Series aligned with `series`.  First `length-1` values are NaN.
    """
    weights = _alma_weights(length, offset, sigma)
    values  = series.to_numpy(dtype=float)
    result  = np.full(len(values), np.nan)

    for i in range(length - 1, len(values)):
        window    = values[i - length + 1 : i + 1]
        result[i] = np.dot(window, weights)

    return pd.Series(result, index=series.index, name=f"alma_{length}")


# ---------------------------------------------------------------------------
# Derived: ATR-normalised slope
# ---------------------------------------------------------------------------

def alma_slope_normalised(
    alma_series: pd.Series,
    atr_series: pd.Series,
) -> pd.Series:
    """
    ATR-normalised first-difference of ALMA.
    slope_norm[t] = (ALMA[t] - ALMA[t-1]) / ATR[t]

    This mirrors the C# logic:
        double rawSlope        = _almaSeries[i] - _almaSeries[i-1];
        double normalizedSlope = rawSlope / currentAtr;
    """
    raw_slope = alma_series.diff(1)
    norm      = raw_slope / atr_series
    return norm.rename("alma_slope_norm")


# ---------------------------------------------------------------------------
# Consecutive bar counter (vectorised)
# ---------------------------------------------------------------------------

def consecutive_bars(condition: pd.Series) -> pd.Series:
    """
    For each bar, count how many *consecutive* True values end at that bar.
    Returns 0 wherever condition is False.

    Example
    -------
    condition = [F, T, T, T, F, T]  →  [0, 1, 2, 3, 0, 1]
    """
    cond = condition.astype(int)
    # Group id resets to 0 whenever condition is False
    group = (cond != cond.shift(1)).cumsum()
    counts = cond.groupby(group).cumsum()
    return counts.where(cond == 1, 0).rename("consecutive_bars")

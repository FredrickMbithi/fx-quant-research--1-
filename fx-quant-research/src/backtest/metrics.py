"""
src/backtest/metrics.py
────────────────────────
Standard performance metrics from trade log and equity curve.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    trade_log: pd.DataFrame,
    equity_curve: pd.Series,
    initial_equity: float,
    periods_per_year: int = 26280,  # 15-min bars per year (365.25 × 24 × 4)
) -> dict:
    """
    Compute a comprehensive performance summary.

    Parameters
    ----------
    trade_log       : DataFrame from PortfolioTracker.trade_log().
    equity_curve    : Series from PortfolioTracker.equity_series().
    initial_equity  : Starting equity.
    periods_per_year: Used for annualisation (15-min bars).

    Returns
    -------
    Dict of metrics.
    """
    if trade_log.empty or equity_curve.empty:
        return {"error": "No trades or equity data."}

    final_equity = equity_curve.iloc[-1]
    n_trades     = len(trade_log)

    # ---- Returns -------------------------------------------------------
    bar_returns  = equity_curve.pct_change().dropna()

    mean_ret   = bar_returns.mean()
    std_ret    = bar_returns.std()
    sharpe     = (mean_ret / std_ret * np.sqrt(periods_per_year)) if std_ret > 0 else 0.0

    total_ret  = (final_equity - initial_equity) / initial_equity
    n_bars     = len(equity_curve)
    years      = n_bars / periods_per_year
    cagr       = (final_equity / initial_equity) ** (1 / max(years, 1e-9)) - 1 if final_equity > 0 else -1.0

    # ---- Drawdown ------------------------------------------------------
    roll_max   = equity_curve.cummax()
    drawdown   = (equity_curve - roll_max) / roll_max
    max_dd     = drawdown.min()

    # ---- Trade stats ---------------------------------------------------
    wins        = trade_log[trade_log["pnl"] > 0]
    losses      = trade_log[trade_log["pnl"] <= 0]
    win_rate    = len(wins) / n_trades if n_trades else 0.0

    avg_win     = wins["pnl"].mean()  if not wins.empty  else 0.0
    avg_loss    = losses["pnl"].mean() if not losses.empty else 0.0

    gross_profit = wins["pnl"].sum()
    gross_loss   = losses["pnl"].sum()
    profit_factor = (gross_profit / abs(gross_loss)) if gross_loss < 0 else np.inf

    avg_bars_held = trade_log["bars_held"].mean()

    # ---- Exit breakdown ------------------------------------------------
    exit_counts = trade_log["exit_reason"].value_counts().to_dict()

    return {
        "n_trades":       n_trades,
        "sharpe":         round(sharpe, 4),
        "cagr":           round(cagr,   4),
        "total_return":   round(total_ret, 4),
        "max_drawdown":   round(max_dd, 4),
        "win_rate":       round(win_rate, 4),
        "profit_factor":  round(profit_factor, 4),
        "avg_win_usd":    round(avg_win, 2),
        "avg_loss_usd":   round(avg_loss, 2),
        "avg_bars_held":  round(avg_bars_held, 1),
        "final_equity":   round(final_equity, 2),
        "gross_profit":   round(gross_profit, 2),
        "gross_loss":     round(gross_loss, 2),
        "exit_breakdown": exit_counts,
    }


def monthly_returns(equity_curve: pd.Series) -> pd.DataFrame:
    """Monthly P&L table useful for withdrawal planning."""
    monthly = equity_curve.resample("ME").last()
    monthly_ret = monthly.pct_change().dropna()
    df = monthly_ret.rename("monthly_return").to_frame()
    df["equity_eom"] = monthly.iloc[1:].values
    return df

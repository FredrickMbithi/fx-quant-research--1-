"""
src/backtest/engine.py
───────────────────────
Event-driven backtesting engine.

Flow per bar
────────────
1. Risk manager: update day / check circuit-breaker.
2. Check open position: SL hit? TP hit? Time-stop?
3. Read signal from strategy (generated on prior bar).
4. If no open position and signal ≠ 0: size → open position.
5. Record equity.

All prices are realistic fills from FillSimulator.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.portfolio.tracker import PortfolioTracker, Position
from src.risk.position_sizing import RiskManager
from src.execution.fills import FillSimulator

log = logging.getLogger(__name__)


def run_backtest(
    df: pd.DataFrame,          # Clean OHLCV with DatetimeIndex
    signals: pd.Series,        # {+1, -1, 0}, same index as df
    features: pd.DataFrame,    # Must contain 'atr' column
    cfg: dict,
) -> tuple[PortfolioTracker, pd.DataFrame]:
    """
    Run the full event-driven backtest.

    Returns
    -------
    (portfolio_tracker, trade_log_df)
    """
    bc   = cfg["backtest"]
    rc   = cfg["risk"]
    pip  = cfg["data"]["pip_size"]

    portfolio = PortfolioTracker(bc["initial_equity"])
    risk_mgr  = RiskManager(cfg)
    filler    = FillSimulator(cfg)

    max_hold  = rc["max_hold_bars"]

    for i, (ts, row) in enumerate(df.iterrows()):
        equity = portfolio.equity

        # 1. Daily circuit-breaker
        risk_mgr.update_day(ts, equity)
        if risk_mgr.check_circuit_breaker(equity):
            if portfolio.position is not None:
                fill = filler.fill_price(row["open"], -portfolio.position.direction)
                portfolio.close_position(ts, fill, "CIRCUIT_BREAKER", i)
            portfolio.record_equity(ts)
            continue

        pos = portfolio.position

        # 2. Manage open position
        if pos is not None:
            low  = row["low"]
            high = row["high"]
            close = row["close"]

            closed = False

            # SL check (worst-case: assume it hits at SL price)
            if pos.direction == 1 and low <= pos.sl:
                portfolio.close_position(ts, pos.sl, "SL", i)
                closed = True
            elif pos.direction == -1 and high >= pos.sl:
                portfolio.close_position(ts, pos.sl, "SL", i)
                closed = True

            # TP check (if not already stopped out)
            if not closed:
                if pos.direction == 1 and high >= pos.tp:
                    portfolio.close_position(ts, pos.tp, "TP", i)
                    closed = True
                elif pos.direction == -1 and low <= pos.tp:
                    portfolio.close_position(ts, pos.tp, "TP", i)
                    closed = True

            # Time-stop
            if not closed and (i - pos.entry_bar) >= max_hold:
                fill = filler.fill_price(row["open"], -pos.direction)
                portfolio.close_position(ts, fill, "TIME", i)
                closed = True

            # Signal reversal — close if signal flips
            sig = signals.iloc[i]
            if not closed and sig != 0 and sig != pos.direction:
                fill = filler.fill_price(row["open"], -pos.direction)
                portfolio.close_position(ts, fill, "SIGNAL_REVERSE", i)
                closed = True

        # 3. Entry
        if portfolio.position is None:
            sig = signals.iloc[i]
            if sig != 0 and i > 0:
                atr_val = features["atr"].iloc[i]
                if np.isnan(atr_val) or atr_val <= 0:
                    portfolio.record_equity(ts)
                    continue

                fill   = filler.fill_price(row["open"], sig)
                units  = risk_mgr.position_size(equity, atr_val)
                comm   = risk_mgr.commission(units)

                if units >= 1.0:  # at least 1 oz (0.01 lot)
                    sl, tp = risk_mgr.sl_tp(fill, sig, atr_val)
                    portfolio.open_position(Position(
                        entry_bar   = i,
                        entry_time  = ts,
                        direction   = sig,
                        units       = units,
                        entry_price = fill,
                        sl          = sl,
                        tp          = tp,
                        commission  = comm,
                    ))

        portfolio.record_equity(ts)

    # Close any open position at end of data
    if portfolio.position is not None:
        last_row = df.iloc[-1]
        last_ts  = df.index[-1]
        fill = filler.fill_price(last_row["close"], -portfolio.position.direction)
        portfolio.close_position(last_ts, fill, "EOD", len(df) - 1)

    return portfolio, portfolio.trade_log()

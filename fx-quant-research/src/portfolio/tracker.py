"""
src/portfolio/tracker.py
─────────────────────────
Tracks equity, open positions, closed trade log, and drawdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd


@dataclass
class Position:
    entry_bar:   int
    entry_time:  pd.Timestamp
    direction:   int            # +1 or -1
    units:       float
    entry_price: float
    sl:          float
    tp:          float
    commission:  float


@dataclass
class ClosedTrade:
    entry_time:  pd.Timestamp
    exit_time:   pd.Timestamp
    direction:   int
    units:       float
    entry_price: float
    exit_price:  float
    sl:          float
    tp:          float
    pnl:         float          # net of commission
    exit_reason: str            # "TP" | "SL" | "TIME" | "SIGNAL" | "EOD"
    bars_held:   int


class PortfolioTracker:
    """
    Records the full portfolio state during backtesting.
    One open position at a time (mirrors the C# bot design).
    """

    def __init__(self, initial_equity: float) -> None:
        self.initial_equity = initial_equity
        self.equity         = initial_equity
        self.position: Optional[Position] = None
        self.trades: list[ClosedTrade] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []

    # ------------------------------------------------------------------

    def open_position(self, pos: Position) -> None:
        assert self.position is None, "Cannot open a second position (single-position mode)."
        self.equity -= pos.commission / 2   # deduct half commission on entry
        self.position = pos

    def close_position(
        self,
        exit_time: pd.Timestamp,
        exit_price: float,
        exit_reason: str,
        bar_index: int,
    ) -> ClosedTrade:
        pos = self.position
        assert pos is not None

        pip_move = (exit_price - pos.entry_price) * pos.direction
        # XAUUSD: P&L = price_diff × direction × units(oz)
        # e.g. long 100 oz, price moves +$2 → P&L = +$200
        pnl_raw  = pip_move * pos.units
        pnl_net  = pnl_raw - pos.commission       # deduct remaining commission on exit

        self.equity += pnl_net

        trade = ClosedTrade(
            entry_time  = pos.entry_time,
            exit_time   = exit_time,
            direction   = pos.direction,
            units       = pos.units,
            entry_price = pos.entry_price,
            exit_price  = exit_price,
            sl          = pos.sl,
            tp          = pos.tp,
            pnl         = pnl_net,
            exit_reason = exit_reason,
            bars_held   = bar_index - pos.entry_bar,
        )
        self.trades.append(trade)
        self.position = None
        return trade

    def record_equity(self, timestamp: pd.Timestamp) -> None:
        self.equity_curve.append((timestamp, self.equity))

    # ------------------------------------------------------------------

    def trade_log(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        return pd.DataFrame([t.__dict__ for t in self.trades])

    def equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        times, values = zip(*self.equity_curve)
        return pd.Series(values, index=times, name="equity")

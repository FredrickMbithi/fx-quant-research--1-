"""
src/risk/position_sizing.py
────────────────────────────
Risk module — decoupled from strategy.

Responsibilities:
  • Translate equity + ATR → position size (units).
  • Compute SL / TP price levels.
  • Evaluate daily circuit-breaker.

Nothing here knows about signals — it only answers "how much?" and "where?".
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class RiskManager:
    """
    Stateful risk manager tracking daily P&L for the circuit-breaker.

    Parameters
    ----------
    cfg : Full config dict.
    """

    def __init__(self, cfg: dict) -> None:
        rc = cfg["risk"]
        ec = cfg["execution"]
        bc = cfg["backtest"]

        self.risk_pct         = rc["risk_pct"] / 100.0
        self.sl_atr_mult      = rc["sl_atr_mult"]
        self.tp_atr_mult      = rc["tp_atr_mult"]
        self.max_hold_bars    = rc["max_hold_bars"]
        self.max_daily_loss   = rc["max_daily_loss_pct"] / 100.0
        self.pip_size         = cfg["data"]["pip_size"]
        self.lot_size         = bc["lot_size"]               # units per standard lot
        self.commission_rt    = ec["commission_per_lot"]     # USD round-turn per lot

        # Circuit-breaker state
        self._day_start_equity: float | None = None
        self._current_day: pd.Timestamp | None = None
        self.halted = False

    # ------------------------------------------------------------------
    # Circuit-breaker
    # ------------------------------------------------------------------

    def update_day(self, timestamp: pd.Timestamp, equity: float) -> None:
        """Call at the start of each bar; resets circuit-breaker daily."""
        day = timestamp.normalize()
        if day != self._current_day:
            self._current_day      = day
            self._day_start_equity = equity
            self.halted            = False

    def check_circuit_breaker(self, equity: float) -> bool:
        """Return True if we've breached the daily loss limit."""
        if self._day_start_equity is None:
            return False
        loss_pct = (self._day_start_equity - equity) / self._day_start_equity
        if loss_pct >= self.max_daily_loss:
            self.halted = True
        return self.halted

    # ------------------------------------------------------------------
    # Position sizing
    # ------------------------------------------------------------------

    def position_size(self, equity: float, atr_value: float) -> float:
        """
        Compute position size in *units* (troy oz) using fixed-fractional ATR sizing.

        For XAUUSD:
          P&L per oz = price_diff  (if price moves $1, you gain/lose $1 per oz)
          Stop distance in price  = ATR × sl_mult
          Risk per oz             = ATR × sl_mult
          Units (oz)              = risk_cash / (ATR × sl_mult)

        We then round down to the nearest 0.01 lot (= 1 oz for 100 oz lots).
        """
        if atr_value <= 0 or np.isnan(atr_value):
            return 0.0

        stop_price_dist = atr_value * self.sl_atr_mult   # USD move per oz
        risk_cash       = equity * self.risk_pct
        units_exact     = risk_cash / stop_price_dist     # oz

        # Floor to nearest 0.01 lot
        lots  = units_exact / self.lot_size
        lots  = max(0.0, np.floor(lots * 100) / 100)
        return lots * self.lot_size

    # ------------------------------------------------------------------
    # SL / TP levels
    # ------------------------------------------------------------------

    def sl_tp(
        self,
        entry_price: float,
        direction: int,          # +1 long, -1 short
        atr_value: float,
    ) -> tuple[float, float]:
        """
        Return (stop_loss_price, take_profit_price).
        Distances are in price units (USD for XAUUSD).
        """
        sl_dist = atr_value * self.sl_atr_mult
        tp_dist = atr_value * self.tp_atr_mult

        if direction == 1:
            sl = entry_price - sl_dist
            tp = entry_price + tp_dist
        else:
            sl = entry_price + sl_dist
            tp = entry_price - tp_dist

        return sl, tp

    # ------------------------------------------------------------------
    # Commission
    # ------------------------------------------------------------------

    def commission(self, units: float) -> float:
        """Round-turn commission in USD. units = oz, lot_size = oz/lot."""
        lots = units / self.lot_size
        return lots * self.commission_rt

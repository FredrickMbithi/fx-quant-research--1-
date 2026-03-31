"""
src/execution/fills.py
───────────────────────
Simulates realistic order fills including spread and slippage.
"""
from __future__ import annotations

import numpy as np


class FillSimulator:
    """
    Converts a signal + bar price into a realistic fill price.

    For a market order on the next bar open:
      Long  fill = open + spread/2 + slippage
      Short fill = open - spread/2 - slippage
    """

    def __init__(self, cfg: dict) -> None:
        ec = cfg["execution"]
        pip = cfg["data"]["pip_size"]
        self.spread_price   = ec["spread_pips"]    * pip
        self.max_slip_price = ec["slippage_pips"]  * pip
        self._rng           = np.random.default_rng(seed=42)

    def fill_price(self, bar_open: float, direction: int) -> float:
        """
        Parameters
        ----------
        bar_open  : Open price of the execution bar.
        direction : +1 (long) or -1 (short).

        Returns
        -------
        Simulated fill price.
        """
        slip = self._rng.uniform(0, self.max_slip_price)
        half_spread = self.spread_price / 2.0

        if direction == 1:
            return bar_open + half_spread + slip
        else:
            return bar_open - half_spread - slip

"""
src/features/microstructure.py
───────────────────────────────
Session-time filters and microstructure helpers.
"""
from __future__ import annotations

import pandas as pd


def session_mask(
    index: pd.DatetimeIndex,
    sessions: dict,
) -> pd.Series:
    """
    Return a boolean Series: True when the bar falls inside an active session.

    Parameters
    ----------
    index    : UTC DatetimeIndex of the bar DataFrame.
    sessions : Dict matching the config schema:
               {
                 "enabled": True,
                 "london":   {"enabled": True, "start": "08:00", "end": "16:30"},
                 "new_york": {"enabled": True, "start": "13:30", "end": "21:00"},
                 "tokyo":    {"enabled": False, ...},
               }
    """
    if not sessions.get("enabled", True):
        return pd.Series(True, index=index, name="in_session")

    active = pd.Series(False, index=index, name="in_session")

    for name, cfg in sessions.items():
        if name == "enabled":
            continue
        if not cfg.get("enabled", False):
            continue

        start = pd.Timestamp(cfg["start"]).time()
        end   = pd.Timestamp(cfg["end"]).time()
        time  = index.time

        if start < end:
            mask = (time >= start) & (time <= end)
        else:                         # overnight session wraps midnight
            mask = (time >= start) | (time <= end)

        active |= mask

    return active

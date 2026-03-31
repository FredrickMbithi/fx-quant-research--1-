"""
experiments/exp_parameter_sweep.py
────────────────────────────────────
Grid search over key ALMA + risk parameters.
Saves a ranked results CSV to /reports/.
"""
from __future__ import annotations

import copy
import json
import logging
import sys
from itertools import product
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loader import load_processed
from src.strategies.hypothesis_strategy import build_features, generate_signals
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


PARAM_GRID = {
    "strategy.alma_length":       [30, 50, 75],
    "strategy.min_slope_atr":     [0.03, 0.05, 0.08],
    "strategy.require_consecutive": [1, 2, 3],
    "risk.sl_atr_mult":           [1.5, 2.0, 2.5],
    "risk.tp_atr_mult":           [3.0, 4.0, 5.0],
}


def set_nested(d: dict, key_path: str, value) -> dict:
    keys = key_path.split(".")
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value
    return d


def main() -> None:
    base_cfg = load_config()
    df = load_processed(
        processed_path=ROOT / base_cfg["data"]["processed_path"],
        raw_path=ROOT / base_cfg["data"]["raw_path"],
    )

    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(product(*values))
    log.info("Running %d parameter combinations …", len(combos))

    rows = []
    for combo in combos:
        cfg = copy.deepcopy(base_cfg)
        param_dict = {}
        for k, v in zip(keys, combo):
            set_nested(cfg, k, v)
            param_dict[k.split(".")[-1]] = v

        try:
            features = build_features(df, cfg)
            signals  = generate_signals(df, cfg)
            portfolio, trade_log = run_backtest(df, signals, features, cfg)
            equity   = portfolio.equity_series()
            m        = compute_metrics(trade_log, equity, base_cfg["backtest"]["initial_equity"])
            row = {**param_dict, **{k: v for k, v in m.items() if k != "exit_breakdown"}}
        except Exception as e:
            row = {**param_dict, "error": str(e)}

        rows.append(row)

    results = pd.DataFrame(rows).sort_values("sharpe", ascending=False)

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    out = reports / "parameter_sweep_results.csv"
    results.to_csv(out, index=False)
    log.info("Sweep results → %s", out)

    print("\nTop 10 parameter combinations by Sharpe:\n")
    print(results.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

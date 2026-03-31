"""
experiments/exp_risk_variation.py
───────────────────────────────────
Test how Sharpe and drawdown respond to different risk% and ATR multiplier settings.
Useful for picking a live risk level consistent with monthly withdrawal targets.
"""
from __future__ import annotations

import copy
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

RISK_LEVELS = [0.25, 0.50, 0.75, 1.0, 1.5, 2.0]
SL_MULTS    = [1.5, 2.0, 2.5, 3.0]


def main() -> None:
    base_cfg = load_config()
    df       = load_processed(
        processed_path=ROOT / base_cfg["data"]["processed_path"],
        raw_path=ROOT / base_cfg["data"]["raw_path"],
    )
    features = build_features(df, base_cfg)
    signals  = generate_signals(df, base_cfg)

    rows = []
    for rp in RISK_LEVELS:
        for sl in SL_MULTS:
            cfg = copy.deepcopy(base_cfg)
            cfg["risk"]["risk_pct"]    = rp
            cfg["risk"]["sl_atr_mult"] = sl
            cfg["risk"]["tp_atr_mult"] = sl * 2  # maintain 1:2 R:R

            portfolio, trade_log = run_backtest(df, signals, features, cfg)
            equity = portfolio.equity_series()
            m      = compute_metrics(trade_log, equity, base_cfg["backtest"]["initial_equity"])
            rows.append({
                "risk_pct": rp, "sl_mult": sl,
                "sharpe": m["sharpe"], "max_dd": m["max_drawdown"],
                "cagr": m["cagr"], "n_trades": m["n_trades"],
            })

    results = pd.DataFrame(rows)
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    results.to_csv(reports / "risk_variation_results.csv", index=False)

    # Pivot heatmap
    pivot = results.pivot(index="risk_pct", columns="sl_mult", values="sharpe")
    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn")
    ax.set_xticks(range(len(SL_MULTS)));  ax.set_xticklabels([f"SL×{s}" for s in SL_MULTS])
    ax.set_yticks(range(len(RISK_LEVELS))); ax.set_yticklabels([f"{r}%" for r in RISK_LEVELS])
    ax.set_title("Sharpe by Risk% and SL Multiplier")
    plt.colorbar(im, ax=ax, label="Sharpe")
    for i in range(len(RISK_LEVELS)):
        for j in range(len(SL_MULTS)):
            ax.text(j, i, f"{pivot.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    plt.tight_layout()
    fig.savefig(reports / "risk_variation_heatmap.png", dpi=150)
    plt.close()
    log.info("Risk variation complete → reports/")
    print(results.sort_values("sharpe", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()

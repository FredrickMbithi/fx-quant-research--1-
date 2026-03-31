"""
experiments/exp_baseline.py
────────────────────────────
Baseline backtest: default parameters from config.yaml.
Outputs trade log, equity curve, and metrics to /reports/.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loader import load_processed
from src.strategies.hypothesis_strategy import build_features, generate_signals
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics, monthly_returns

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


def main() -> None:
    cfg = load_config()

    # ── Data ──────────────────────────────────────────────────────────────
    log.info("Loading data …")
    df = load_processed(
        processed_path=ROOT / cfg["data"]["processed_path"],
        raw_path=ROOT / cfg["data"]["raw_path"],
    )
    log.info("Data: %d bars  [%s → %s]", len(df), df.index[0].date(), df.index[-1].date())

    # ── Features & Signals ────────────────────────────────────────────────
    log.info("Building features …")
    features = build_features(df, cfg)

    log.info("Generating signals …")
    signals  = generate_signals(df, cfg)

    long_count  = (signals ==  1).sum()
    short_count = (signals == -1).sum()
    log.info("Signals: %d long, %d short", long_count, short_count)

    # ── Backtest ──────────────────────────────────────────────────────────
    log.info("Running backtest …")
    portfolio, trade_log = run_backtest(df, signals, features, cfg)

    equity = portfolio.equity_series()
    metrics = compute_metrics(trade_log, equity, cfg["backtest"]["initial_equity"])

    # ── Reports ───────────────────────────────────────────────────────────
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)

    # Trade log
    if not trade_log.empty:
        trade_log.to_csv(reports / "baseline_trade_log.csv", index=False)
        log.info("Trade log → reports/baseline_trade_log.csv")

    # Metrics JSON
    (reports / "baseline_metrics.json").write_text(json.dumps(metrics, indent=2))
    log.info("Metrics → reports/baseline_metrics.json")

    # Equity curve
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1.2, 1]})

    axes[0].plot(equity.index, equity.values, linewidth=1, color="#2196F3")
    axes[0].set_title("XAUUSD 15min — ALMA Trend Strategy (Baseline)", fontsize=13)
    axes[0].set_ylabel("Equity (USD)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(cfg["backtest"]["initial_equity"], color="grey", linestyle="--", linewidth=0.8)

    # Drawdown subplot
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max * 100
    axes[1].fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.5)
    axes[1].set_ylabel("Drawdown (%)")
    axes[1].grid(True, alpha=0.3)

    # Monthly returns bar chart
    mon_ret = monthly_returns(equity)["monthly_return"] * 100
    colors  = ["#4CAF50" if v >= 0 else "#F44336" for v in mon_ret.values]
    axes[2].bar(range(len(mon_ret)), mon_ret.values, color=colors, width=0.8)
    axes[2].set_xticks(range(len(mon_ret)))
    axes[2].set_xticklabels(
        [ts.strftime("%Y-%m") for ts in mon_ret.index],
        rotation=90, fontsize=6,
    )
    axes[2].set_ylabel("Monthly Return (%)")
    axes[2].axhline(0, color="black", linewidth=0.8)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(reports / "baseline_equity_curve.png", dpi=150)
    plt.close()
    log.info("Equity curve → reports/baseline_equity_curve.png")

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "═" * 50)
    print("  BASELINE BACKTEST RESULTS")
    print("═" * 50)
    for k, v in metrics.items():
        if k == "exit_breakdown":
            print(f"  {'Exit breakdown':<22}: {v}")
        else:
            print(f"  {k:<22}: {v}")
    print("═" * 50 + "\n")


if __name__ == "__main__":
    main()

"""
experiments/exp_hmm_regime.py
──────────────────────────────
Compare baseline ALMA strategy vs HMM-gated version.

Outputs to reports/:
  • hmm_regime_series.parquet       — bar-by-bar regime labels
  • hmm_comparison_metrics.json     — side-by-side performance numbers
  • hmm_equity_comparison.png       — equity curves overlay
  • hmm_regime_map.png              — price coloured by regime + signals
  • hmm_trade_log.csv               — HMM strategy trade log
  • hmm_monthly_comparison.png      — monthly returns bar chart comparison
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loader import load_processed
from src.strategies.hypothesis_strategy import build_features, generate_signals
from src.strategies.hmm_strategy import build_features_hmm, generate_signals_hmm
from src.backtest.engine import run_backtest
from src.backtest.metrics import compute_metrics, monthly_returns
from src.features.regime import TRENDING, RANGING

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

REPORTS = ROOT / "reports"
REPORTS.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# 1. Load data
# ══════════════════════════════════════════════════════════════════════════════

cfg = load_config()
log.info("Loading data …")
df = load_processed(
    processed_path=ROOT / cfg["data"]["processed_path"],
    raw_path=ROOT       / cfg["data"]["raw_path"],
)
log.info("Data: %d bars  [%s → %s]", len(df), df.index[0].date(), df.index[-1].date())


# ══════════════════════════════════════════════════════════════════════════════
# 2. Baseline (no HMM)
# ══════════════════════════════════════════════════════════════════════════════

log.info("Running baseline strategy …")
base_features = build_features(df, cfg)
base_signals  = generate_signals(df, cfg)
base_port, base_trades = run_backtest(df, base_signals, base_features, cfg)
base_equity   = base_port.equity_series()
base_metrics  = compute_metrics(base_trades, base_equity, cfg["backtest"]["initial_equity"])


# ══════════════════════════════════════════════════════════════════════════════
# 3. HMM-gated strategy
# ══════════════════════════════════════════════════════════════════════════════

log.info("Running HMM regime filter + strategy …")
hmm_features = build_features_hmm(df, cfg)                         # single HMM fit
hmm_signals  = generate_signals_hmm(df, cfg, features=hmm_features) # reuse features
hmm_port, hmm_trades = run_backtest(df, hmm_signals, hmm_features, cfg)
hmm_equity   = hmm_port.equity_series()
hmm_metrics  = compute_metrics(hmm_trades, hmm_equity, cfg["backtest"]["initial_equity"])

regime = hmm_features["regime"]


# ══════════════════════════════════════════════════════════════════════════════
# 4. Save artefacts
# ══════════════════════════════════════════════════════════════════════════════

regime.to_frame().to_parquet(REPORTS / "hmm_regime_series.parquet")
if not hmm_trades.empty:
    hmm_trades.to_csv(REPORTS / "hmm_trade_log.csv", index=False)

comparison = {
    "baseline": {k: v for k, v in base_metrics.items() if k != "exit_breakdown"},
    "hmm_gated": {k: v for k, v in hmm_metrics.items() if k != "exit_breakdown"},
    "baseline_exits":  base_metrics.get("exit_breakdown", {}),
    "hmm_exits":       hmm_metrics.get("exit_breakdown", {}),
}
(REPORTS / "hmm_comparison_metrics.json").write_text(json.dumps(comparison, indent=2))


# ══════════════════════════════════════════════════════════════════════════════
# 5. Regime statistics
# ══════════════════════════════════════════════════════════════════════════════

valid_regime = regime.dropna()
n_total      = len(valid_regime)
n_trending   = (valid_regime == TRENDING).sum()
n_ranging    = (valid_regime == RANGING).sum()
pct_trending = 100 * n_trending / n_total
pct_ranging  = 100 * n_ranging  / n_total

# Signal filtering effect
base_sig_count = (base_signals != 0).sum()
hmm_sig_count  = (hmm_signals  != 0).sum()
filter_rate    = 100 * (1 - hmm_sig_count / base_sig_count) if base_sig_count else 0

print("\n" + "═" * 60)
print("  REGIME STATISTICS")
print("═" * 60)
print(f"  TRENDING bars : {n_trending:>7,}  ({pct_trending:.1f}%)")
print(f"  RANGING  bars : {n_ranging:>7,}  ({pct_ranging:.1f}%)")
print(f"  HMM filtered  : {base_sig_count - hmm_sig_count:>7,} signals suppressed ({filter_rate:.1f}%)")

print("\n" + "═" * 60)
print("  PERFORMANCE COMPARISON")
print("═" * 60)
keys = ["n_trades", "sharpe", "cagr", "max_drawdown", "win_rate",
        "profit_factor", "avg_win_usd", "avg_loss_usd", "final_equity"]
print(f"  {'Metric':<22}  {'Baseline':>12}  {'HMM-Gated':>12}  {'Delta':>10}")
print("  " + "─" * 60)
for k in keys:
    b = base_metrics.get(k, "—")
    h = hmm_metrics.get(k, "—")
    try:
        delta = h - b
        delta_str = f"{delta:+.4f}" if abs(delta) < 1000 else f"{delta:+.1f}"
    except TypeError:
        delta_str = "—"
    print(f"  {k:<22}  {str(b):>12}  {str(h):>12}  {delta_str:>10}")
print("═" * 60 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# 6. Plot 1 — Equity curve comparison
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 1, figsize=(15, 12),
                         gridspec_kw={"height_ratios": [3, 1.2, 1.2]}, sharex=False)

ax = axes[0]
ax.plot(base_equity.index, base_equity.values, color="#90A4AE", linewidth=0.9,
        label=f"Baseline  Sharpe={base_metrics['sharpe']:.3f}  MaxDD={base_metrics['max_drawdown']:.1%}")
ax.plot(hmm_equity.index,  hmm_equity.values,  color="#1565C0", linewidth=1.4,
        label=f"HMM-Gated Sharpe={hmm_metrics['sharpe']:.3f}  MaxDD={hmm_metrics['max_drawdown']:.1%}")
ax.axhline(cfg["backtest"]["initial_equity"], color="grey", linestyle="--", linewidth=0.7)
ax.set_title("XAUUSD 15min — ALMA Trend: Baseline vs HMM Regime Filter", fontsize=13)
ax.set_ylabel("Equity (USD)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Drawdown subplot
def calc_dd(eq):
    return (eq - eq.cummax()) / eq.cummax() * 100

ax2 = axes[1]
ax2.fill_between(base_equity.index, calc_dd(base_equity), 0, color="#B0BEC5", alpha=0.6, label="Baseline DD")
ax2.fill_between(hmm_equity.index,  calc_dd(hmm_equity),  0, color="#F44336", alpha=0.5, label="HMM DD")
ax2.set_ylabel("Drawdown (%)")
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Regime over time (fraction trending per month)
ax3 = axes[2]
monthly_trend = (regime == TRENDING).resample("ME").mean() * 100
ax3.fill_between(monthly_trend.index, monthly_trend.values, color="#2E7D32", alpha=0.6)
ax3.axhline(50, color="black", linestyle="--", linewidth=0.7)
ax3.set_ylabel("% Trending (monthly)")
ax3.set_ylim(0, 100)
ax3.grid(True, alpha=0.3)
ax3.set_xlabel("Date")

plt.tight_layout()
fig.savefig(REPORTS / "hmm_equity_comparison.png", dpi=150)
plt.close()
log.info("Saved hmm_equity_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. Plot 2 — Regime map (500-bar sample)
# ══════════════════════════════════════════════════════════════════════════════

# Find a representative sample with regime transitions
regime_valid = regime.dropna()
# Use bars from 2022 onwards for recency
sample_start = pd.Timestamp("2022-01-01", tz="UTC")
mask_2022    = df.index >= sample_start
idx_start    = int(np.argmax(mask_2022))
WINDOW       = 800

s_df   = df.iloc[idx_start : idx_start + WINDOW]
s_feat = hmm_features.iloc[idx_start : idx_start + WINDOW]
s_sig  = hmm_signals.iloc[idx_start : idx_start + WINDOW]
s_reg  = s_feat["regime"]

fig, axes = plt.subplots(3, 1, figsize=(16, 11),
                         gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)

ax = axes[0]
# Background shading by regime
for i in range(len(s_reg) - 1):
    if pd.isna(s_reg.iloc[i]):
        continue
    color = "#E8F5E9" if s_reg.iloc[i] == TRENDING else "#FCE4EC"
    ax.axvspan(s_df.index[i], s_df.index[i+1], color=color, alpha=0.4, linewidth=0)

ax.plot(s_df.index, s_df["close"], color="#37474F", linewidth=0.9, label="Close")
ax.plot(s_df.index, s_feat["alma"], color="#1565C0", linewidth=1.4, label="ALMA(50)")

# Signal markers — only HMM-gated
hmm_longs  = s_sig[s_sig ==  1].index
hmm_shorts = s_sig[s_sig == -1].index
ax.scatter(hmm_longs,  s_df.loc[hmm_longs,  "close"], marker="^", color="#2E7D32", s=50, zorder=6, label="Long (HMM)")
ax.scatter(hmm_shorts, s_df.loc[hmm_shorts, "close"], marker="v", color="#C62828", s=50, zorder=6, label="Short (HMM)")

trend_patch  = mpatches.Patch(color="#E8F5E9", alpha=0.8, label="TRENDING regime")
range_patch  = mpatches.Patch(color="#FCE4EC", alpha=0.8, label="RANGING regime")
ax.legend(handles=[trend_patch, range_patch] + ax.get_legend_handles_labels()[0][1:],
          fontsize=8, loc="upper left")
ax.set_title("HMM Regime Map — XAUUSD 15min (800-bar sample, 2022+)", fontsize=12)
ax.set_ylabel("Price (USD)")
ax.grid(True, alpha=0.25)

# ALMA slope
ax2 = axes[1]
threshold = cfg["strategy"]["min_slope_atr"]
ax2.plot(s_df.index, s_feat["alma_slope_norm"], color="#7B1FA2", linewidth=0.8)
ax2.axhline( threshold, color="#2E7D32", linestyle="--", linewidth=0.8)
ax2.axhline(-threshold, color="#C62828", linestyle="--", linewidth=0.8)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("Slope / ATR")
ax2.grid(True, alpha=0.25)

# Regime series
ax3 = axes[2]
reg_numeric = s_reg.fillna(-1)
ax3.fill_between(s_df.index, reg_numeric.values, 0,
                 where=reg_numeric.values == TRENDING,
                 color="#2E7D32", alpha=0.7, label="TRENDING")
ax3.fill_between(s_df.index, 1, reg_numeric.values,
                 where=reg_numeric.values == RANGING,
                 color="#EF9A9A", alpha=0.7, label="RANGING")
ax3.set_yticks([0, 1]); ax3.set_yticklabels(["RANGING", "TRENDING"])
ax3.set_ylabel("Regime")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.25)

plt.tight_layout()
fig.savefig(REPORTS / "hmm_regime_map.png", dpi=150)
plt.close()
log.info("Saved hmm_regime_map.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. Plot 3 — Monthly returns comparison
# ══════════════════════════════════════════════════════════════════════════════

base_mon = monthly_returns(base_equity)["monthly_return"] * 100
hmm_mon  = monthly_returns(hmm_equity)["monthly_return"]  * 100

# Align on common months
common = base_mon.index.intersection(hmm_mon.index)
base_mon = base_mon.loc[common]
hmm_mon  = hmm_mon.loc[common]

x     = np.arange(len(common))
width = 0.38
labels_x = [ts.strftime("%Y-%m") for ts in common]

fig, ax = plt.subplots(figsize=(18, 6))
ax.bar(x - width/2, base_mon.values, width, label="Baseline",
       color=["#81C784" if v >= 0 else "#E57373" for v in base_mon.values], alpha=0.7)
ax.bar(x + width/2, hmm_mon.values,  width, label="HMM-Gated",
       color=["#1B5E20" if v >= 0 else "#B71C1C" for v in hmm_mon.values], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels_x, rotation=90, fontsize=6.5)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Monthly Return (%)")
ax.set_title("Monthly Returns: Baseline vs HMM-Gated Strategy", fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
fig.savefig(REPORTS / "hmm_monthly_comparison.png", dpi=150)
plt.close()
log.info("Saved hmm_monthly_comparison.png")

log.info("All outputs → reports/")

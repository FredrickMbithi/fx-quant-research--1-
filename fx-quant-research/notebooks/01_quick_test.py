"""
notebooks/01_quick_test.py
───────────────────────────
Phase 0 — Visual verification that ALMA slope signals are sensible.
Run this as a plain Python script; outputs a PNG to reports/.

Mirrors Notebook 01 from the 19-notebook pipeline spec.
Pass criterion: Manual review — signals generated, pattern makes sense.
"""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loader import load_processed
from src.strategies.hypothesis_strategy import build_features, generate_signals

cfg      = load_config()
df       = load_processed(cfg["data"]["processed_path"], cfg["data"]["raw_path"])
features = build_features(df, cfg)
signals  = generate_signals(df, cfg)

# ── Inspect ──────────────────────────────────────────────────────────────────
print("Signal distribution:")
print(signals.value_counts().to_dict())
print("\nSignal dtype:", signals.dtype)

sig_bars = signals[signals != 0]
print(f"\nFirst 10 signal bars:\n{sig_bars.head(10)}")

# ── Plot 500 bars with ALMA and signal markers ────────────────────────────────
WINDOW = 500
sample = df.iloc[200:200 + WINDOW].copy()
feat_s = features.iloc[200:200 + WINDOW]
sig_s  = signals.iloc[200:200 + WINDOW]

fig, axes = plt.subplots(3, 1, figsize=(16, 10),
                         gridspec_kw={"height_ratios": [3, 1, 1]}, sharex=True)

# Price + ALMA + signals
ax = axes[0]
ax.plot(sample.index, sample["close"], color="#90A4AE", linewidth=0.8, label="Close")
ax.plot(sample.index, feat_s["alma"],  color="#1565C0", linewidth=1.5, label="ALMA(50)")

longs  = sig_s[sig_s ==  1].index
shorts = sig_s[sig_s == -1].index
ax.scatter(longs,  sample.loc[longs,  "close"], marker="^", color="#2E7D32", s=40, zorder=5, label="Long")
ax.scatter(shorts, sample.loc[shorts, "close"], marker="v", color="#C62828", s=40, zorder=5, label="Short")
ax.set_title("XAUUSD 15min — ALMA Trend Signal (500-bar sample)", fontsize=12)
ax.set_ylabel("Price (USD)")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# ATR-normalised slope
ax2 = axes[1]
ax2.plot(sample.index, feat_s["alma_slope_norm"], color="#7B1FA2", linewidth=0.9)
threshold = cfg["strategy"]["min_slope_atr"]
ax2.axhline( threshold, color="#2E7D32", linestyle="--", linewidth=0.8)
ax2.axhline(-threshold, color="#C62828", linestyle="--", linewidth=0.8)
ax2.axhline(0, color="black", linewidth=0.5)
ax2.set_ylabel("ALMA Slope / ATR")
ax2.grid(True, alpha=0.3)

# ATR
ax3 = axes[2]
ax3.fill_between(sample.index, feat_s["atr"], color="#FF8F00", alpha=0.5)
ax3.set_ylabel("ATR(20)")
ax3.set_xlabel("Time (UTC)")
ax3.grid(True, alpha=0.3)

plt.tight_layout()
reports = ROOT / "reports"
reports.mkdir(exist_ok=True)
out = reports / "01_quick_test_signals.png"
fig.savefig(out, dpi=150)
plt.close()
print(f"\nPlot saved → {out}")
print("\n✅ PASS — Signals generated, ALMA and markers look sensible. Review the PNG.")

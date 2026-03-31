"""
notebooks/02_univariate_ic_test.py
────────────────────────────────────
Phase 1 — Test predictive power of the ALMA slope signal at multiple forward horizons.

Pass criterion: |IC| > 0.03 AND |t-stat| > 2.0 at best horizon.
Outputs: reports/02_ic_results.csv, reports/02_ic_analysis.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loader import load_processed
from src.strategies.hypothesis_strategy import build_features

cfg      = load_config()
df       = load_processed(cfg["data"]["processed_path"], cfg["data"]["raw_path"])
features = build_features(df, cfg)

# Raw signal (pre-shift) for IC — we want correlation at generation bar
signal = features["alma_slope_norm"].copy()

HORIZONS = [1, 2, 4, 6, 12, 24]   # bars (15-min each)

rows = []
for H in HORIZONS:
    fwd_ret = df["close"].pct_change(H).shift(-H)   # forward return over H bars
    valid   = signal.notna() & fwd_ret.notna()
    s_vals  = signal[valid].values
    r_vals  = fwd_ret[valid].values

    rho, p_spearman = stats.spearmanr(s_vals, r_vals)
    t_stat  = rho * np.sqrt((valid.sum() - 2) / (1 - rho**2 + 1e-12))
    hit     = np.mean(np.sign(s_vals) == np.sign(r_vals))

    # Long / short breakdown
    long_mask  = s_vals > 0
    short_mask = s_vals < 0
    hit_long   = np.mean(r_vals[long_mask]  > 0) if long_mask.any()  else np.nan
    hit_short  = np.mean(r_vals[short_mask] < 0) if short_mask.any() else np.nan

    rows.append({
        "horizon":    H,
        "horizon_h":  H * 0.25,
        "ic":         round(rho, 5),
        "t_stat":     round(t_stat, 3),
        "p_value":    round(p_spearman, 5),
        "hit_rate":   round(hit, 4),
        "hit_long":   round(hit_long, 4),
        "hit_short":  round(hit_short, 4),
        "n":          int(valid.sum()),
        "pass":       abs(rho) > 0.03 and abs(t_stat) > 2.0,
    })

ic_df = pd.DataFrame(rows)
print(ic_df.to_string(index=False))

best = ic_df.loc[ic_df["ic"].abs().idxmax()]
print(f"\nBest horizon: {int(best['horizon'])} bars ({best['horizon_h']:.2f}h)")
print(f"  IC={best['ic']:.5f}, t={best['t_stat']:.3f}, hit={best['hit_rate']:.4f}")
print(f"  PASS: {best['pass']}")

# ── Save ─────────────────────────────────────────────────────────────────────
reports = ROOT / "reports"
reports.mkdir(exist_ok=True)
ic_df.to_csv(reports / "02_ic_results.csv", index=False)

decision = {
    "best_horizon": int(best["horizon"]),
    "ic":           float(best["ic"]),
    "t_stat":       float(best["t_stat"]),
    "pass":         bool(best["pass"]),
}
(reports / "ic_decision.json").write_text(json.dumps(decision, indent=2))

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
labels = [f"{H}b\n({H*0.25:.1f}h)" for H in HORIZONS]

# IC by horizon
ax = axes[0, 0]
colors = ["#2E7D32" if v > 0 else "#C62828" for v in ic_df["ic"]]
ax.bar(labels, ic_df["ic"], color=colors)
ax.axhline(0.03, color="green", linestyle="--", linewidth=0.8, label="+0.03 threshold")
ax.axhline(-0.03, color="red", linestyle="--", linewidth=0.8, label="-0.03 threshold")
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("Spearman IC by Horizon")
ax.set_ylabel("IC")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# t-stat by horizon
ax = axes[0, 1]
ax.bar(labels, ic_df["t_stat"].abs(), color="#1565C0")
ax.axhline(2.0, color="orange", linestyle="--", linewidth=1.0, label="|t|=2.0")
ax.set_title("|t-stat| by Horizon")
ax.set_ylabel("|t-stat|")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")

# Hit rate
ax = axes[1, 0]
ax.plot(labels, ic_df["hit_rate"],  marker="o", label="Overall",   color="#1565C0")
ax.plot(labels, ic_df["hit_long"],  marker="s", label="Long",      color="#2E7D32")
ax.plot(labels, ic_df["hit_short"], marker="^", label="Short",     color="#C62828")
ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.8)
ax.set_title("Hit Rate by Horizon")
ax.set_ylabel("Hit Rate")
ax.legend(fontsize=8)
ax.set_ylim(0.4, 0.65)
ax.grid(True, alpha=0.3)

# Return distribution at best horizon
ax = axes[1, 1]
H_best  = int(best["horizon"])
fwd_ret = df["close"].pct_change(H_best).shift(-H_best)
valid   = signal.notna() & fwd_ret.notna()
s_vals  = signal[valid].values
r_vals  = fwd_ret[valid].values

long_ret  = r_vals[s_vals > 0] * 100
short_ret = -r_vals[s_vals < 0] * 100   # flip sign so positive = correct short

ax.hist(long_ret,  bins=80, alpha=0.5, color="#2E7D32", label=f"Long (n={len(long_ret):,})",  density=True)
ax.hist(short_ret, bins=80, alpha=0.5, color="#C62828", label=f"Short (n={len(short_ret):,})", density=True)
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title(f"Return Distribution at Horizon {H_best}b")
ax.set_xlabel("Forward Return (%)")
ax.legend(fontsize=8)
ax.set_xlim(-1, 1)
ax.grid(True, alpha=0.3)

plt.suptitle("Univariate IC Analysis — ALMA Slope (XAUUSD 15min)", fontsize=13)
plt.tight_layout()
fig.savefig(reports / "02_ic_analysis.png", dpi=150)
plt.close()
print(f"\nOutputs → reports/02_ic_results.csv, reports/02_ic_analysis.png")

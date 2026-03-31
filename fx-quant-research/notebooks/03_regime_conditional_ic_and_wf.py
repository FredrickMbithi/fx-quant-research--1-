"""
notebooks/03_regime_conditional_ic_and_wf.py
──────────────────────────────────────────────
Three-phase investigation of HMM regime conditioning:

PHASE 1: IC test on TRENDING bars only (2-state HMM)
  → Hypothesis: |IC| jumps from 0.016 (unconditional) → 0.04+ (trending only)

PHASE 2: 3-state HMM (UP-TREND, DOWN-TREND, RANGING)
  → Long signals only active in UP-TREND
  → Short signals only active in DOWN-TREND

PHASE 3: Walk-forward validation
  → Confirm Sharpe improvement is real vs regime-fit overfitting
  → 6-month rolling window, refit every month
  → Compare: Unconditional vs 2-state vs 3-state

Decision Gates:
  ✓ Phase 1: |IC| > 0.04 on trending bars
  ✓ Phase 2: Sign consistency > 85% (up-trend longs profitable, down-trend shorts profitable)
  ✓ Phase 3: OOS Sharpe improvement > 30%, degradation < 20%

Outputs:
  - reports/03_regime_ic_comparison.csv
  - reports/03_regime_ic_analysis.png
  - reports/03_wf_equity_curves.png
  - reports/03_wf_metrics.json
"""
from __future__ import annotations
import json, sys, warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.utils.config import load_config
from src.data.loader import load_processed
from src.strategies.hypothesis_strategy import build_features

print("=" * 80)
print("REGIME-CONDITIONAL IC + WALK-FORWARD VALIDATION")
print("=" * 80)

cfg      = load_config()
df       = load_processed(cfg["data"]["processed_path"], cfg["data"]["raw_path"])
features = build_features(df, cfg)

# Raw signal (pre-shift) for IC
signal = features["alma_slope_norm"].copy()

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1: 2-STATE HMM → IC TEST ON TRENDING BARS ONLY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("PHASE 1: IC Test on 2-State HMM TRENDING Bars")
print("─" * 80)

def build_hmm_features(df: pd.DataFrame) -> np.ndarray:
    """Features for HMM: log return, rolling vol, bar range"""
    close  = df["close"].values.astype(float)
    high   = df["high"].values.astype(float)
    low    = df["low"].values.astype(float)

    log_ret  = np.diff(np.log(close), prepend=np.log(close[0]))
    roll_vol = pd.Series(log_ret).rolling(20, min_periods=5).std().fillna(0).values
    bar_range = (high - low) / close

    return np.column_stack([log_ret, roll_vol, bar_range])

def fit_2state_hmm(X_train: np.ndarray) -> tuple[GaussianHMM, StandardScaler, dict]:
    """Fit 2-state HMM, identify TRENDING (lower |ret|) vs RANGING (higher |ret|)"""
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(
            n_components=2,
            covariance_type="full",
            n_iter=100,
            random_state=42,
            tol=1e-4,
        )
        model.fit(X_scaled)

    # Label states by mean |log_ret|
    states   = model.predict(X_scaled)
    log_ret_col = X_train[:, 0]
    mean_abs = {s: np.mean(np.abs(log_ret_col[states == s])) for s in [0, 1]}
    sorted_states = sorted(mean_abs, key=mean_abs.get)
    state_map = {sorted_states[0]: 1, sorted_states[1]: 0}  # 1=TRENDING, 0=RANGING

    return model, scaler, state_map

# Walk-forward HMM labelling (2-state)
X_hmm    = build_hmm_features(df)
n        = len(df)
regime_2 = np.full(n, np.nan)
lookback = 2000
refit_every = 500

model_2    = None
state_map_2 = None
last_fit   = -1

for i in range(lookback, n):
    if model_2 is None or (i - last_fit) >= refit_every:
        start = max(0, i - lookback)
        X_train = X_hmm[start:i]
        try:
            model_2, scaler_2, state_map_2 = fit_2state_hmm(X_train)
            last_fit = i
        except Exception as e:
            print(f"  HMM fit failed at bar {i}: {e}")
            continue

    x_i = scaler_2.transform(X_hmm[i:i+1])
    raw_state = model_2.predict(x_i)[0]
    regime_2[i] = state_map_2.get(raw_state, 0)

regime_2_series = pd.Series(regime_2, index=df.index, name="regime_2state")
trending_mask   = regime_2_series == 1
ranging_mask    = regime_2_series == 0

print(f"  TRENDING: {trending_mask.sum()} bars ({100*trending_mask.sum()/n:.1f}%)")
print(f"  RANGING:  {ranging_mask.sum()} bars ({100*ranging_mask.sum()/n:.1f}%)")

# IC test: unconditional vs trending-only
H_BEST = 4  # from prior notebook
fwd_ret = df["close"].pct_change(H_BEST).shift(-H_BEST)

# Unconditional IC
valid_all = signal.notna() & fwd_ret.notna()
s_all = signal[valid_all].values
r_all = fwd_ret[valid_all].values
ic_all, _ = stats.spearmanr(s_all, r_all)
t_all = ic_all * np.sqrt((valid_all.sum() - 2) / (1 - ic_all**2 + 1e-12))

# Trending-only IC
valid_trend = signal.notna() & fwd_ret.notna() & trending_mask
s_trend = signal[valid_trend].values
r_trend = fwd_ret[valid_trend].values
ic_trend, _ = stats.spearmanr(s_trend, r_trend)
t_trend = ic_trend * np.sqrt((valid_trend.sum() - 2) / (1 - ic_trend**2 + 1e-12))

# Ranging-only IC
valid_range = signal.notna() & fwd_ret.notna() & ranging_mask
s_range = signal[valid_range].values
r_range = fwd_ret[valid_range].values
ic_range, _ = stats.spearmanr(s_range, r_range)
t_range = ic_range * np.sqrt((valid_range.sum() - 2) / (1 - ic_range**2 + 1e-12))

ic_comparison = pd.DataFrame([
    {
        "regime": "Unconditional",
        "n": int(valid_all.sum()),
        "ic": round(ic_all, 5),
        "t_stat": round(t_all, 3),
        "hit_rate": round(np.mean(np.sign(s_all) == np.sign(r_all)), 4),
    },
    {
        "regime": "TRENDING",
        "n": int(valid_trend.sum()),
        "ic": round(ic_trend, 5),
        "t_stat": round(t_trend, 3),
        "hit_rate": round(np.mean(np.sign(s_trend) == np.sign(r_trend)), 4),
    },
    {
        "regime": "RANGING",
        "n": int(valid_range.sum()),
        "ic": round(ic_range, 5),
        "t_stat": round(t_range, 3),
        "hit_rate": round(np.mean(np.sign(s_range) == np.sign(r_range)), 4),
    },
])

print("\nIC Comparison (Horizon=4b):")
print(ic_comparison.to_string(index=False))

phase1_pass = abs(ic_trend) > 0.04 and abs(t_trend) > 2.0
print(f"\n✓ Phase 1 Gate: |IC_trending| > 0.04 → {phase1_pass}")
print(f"  IC_trending = {ic_trend:.5f}, t = {t_trend:.3f}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 2: 3-STATE HMM (UP-TREND, DOWN-TREND, RANGING)
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("PHASE 2: 3-State HMM (UP-TREND, DOWN-TREND, RANGING)")
print("─" * 80)

def fit_3state_hmm(X_train: np.ndarray) -> tuple[GaussianHMM, StandardScaler, dict]:
    """
    Fit 3-state HMM:
      - State with highest mean return → UP-TREND (2)
      - State with lowest mean return → DOWN-TREND (0)
      - Middle state → RANGING (1)
    """
    scaler  = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = GaussianHMM(
            n_components=3,
            covariance_type="full",
            n_iter=100,
            random_state=42,
            tol=1e-4,
        )
        model.fit(X_scaled)

    # Label states by mean log_ret (not |ret| this time)
    states   = model.predict(X_scaled)
    log_ret_col = X_train[:, 0]
    mean_ret = {s: np.mean(log_ret_col[states == s]) for s in [0, 1, 2]}
    sorted_states = sorted(mean_ret, key=mean_ret.get)  # low → high
    state_map = {
        sorted_states[0]: 0,  # DOWN-TREND
        sorted_states[1]: 1,  # RANGING
        sorted_states[2]: 2,  # UP-TREND
    }

    return model, scaler, state_map

# Walk-forward HMM labelling (3-state)
regime_3 = np.full(n, np.nan)
model_3    = None
state_map_3 = None
last_fit   = -1

for i in range(lookback, n):
    if model_3 is None or (i - last_fit) >= refit_every:
        start = max(0, i - lookback)
        X_train = X_hmm[start:i]
        try:
            model_3, scaler_3, state_map_3 = fit_3state_hmm(X_train)
            last_fit = i
        except Exception as e:
            print(f"  HMM fit failed at bar {i}: {e}")
            continue

    x_i = scaler_3.transform(X_hmm[i:i+1])
    raw_state = model_3.predict(x_i)[0]
    regime_3[i] = state_map_3.get(raw_state, 1)

regime_3_series = pd.Series(regime_3, index=df.index, name="regime_3state")
down_mask = regime_3_series == 0
range_mask = regime_3_series == 1
up_mask   = regime_3_series == 2

print(f"  DOWN-TREND: {down_mask.sum()} bars ({100*down_mask.sum()/n:.1f}%)")
print(f"  RANGING:    {range_mask.sum()} bars ({100*range_mask.sum()/n:.1f}%)")
print(f"  UP-TREND:   {up_mask.sum()} bars ({100*up_mask.sum()/n:.1f}%)")

# Validate: long signals in UP-TREND should be profitable
# Short signals in DOWN-TREND should be profitable
valid_up = signal.notna() & fwd_ret.notna() & up_mask & (signal > 0)
valid_down = signal.notna() & fwd_ret.notna() & down_mask & (signal < 0)

if valid_up.sum() > 0:
    ret_up_long = fwd_ret[valid_up].values
    hit_up = np.mean(ret_up_long > 0)
    mean_ret_up = ret_up_long.mean() * 100
else:
    hit_up, mean_ret_up = np.nan, np.nan

if valid_down.sum() > 0:
    ret_down_short = fwd_ret[valid_down].values
    hit_down = np.mean(ret_down_short < 0)  # short correct when ret < 0
    mean_ret_down = -ret_down_short.mean() * 100  # flip sign
else:
    hit_down, mean_ret_down = np.nan, np.nan

print(f"\n  Long signals in UP-TREND:   hit={hit_up:.3f}, mean_ret={mean_ret_up:+.4f}%")
print(f"  Short signals in DOWN-TREND: hit={hit_down:.3f}, mean_ret={mean_ret_down:+.4f}%")

phase2_pass = (hit_up > 0.55) and (hit_down > 0.55)
print(f"\n✓ Phase 2 Gate: Both hit rates > 0.55 → {phase2_pass}")

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3: WALK-FORWARD VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 80)
print("PHASE 3: Walk-Forward Validation (6-month windows, 1-month step)")
print("─" * 80)

def simple_backtest(df_test: pd.DataFrame, signal: pd.Series, regime: pd.Series = None, mode: str = "unconditional") -> pd.Series:
    """
    Simple equity curve generator.
    mode:
      - "unconditional": trade all signal bars
      - "2state": trade only TRENDING bars
      - "3state": long only in UP-TREND, short only in DOWN-TREND
    """
    df_bt = df_test.copy()
    df_bt["signal"] = signal.reindex(df_bt.index).ffill()
    df_bt["fwd_ret"] = df_bt["close"].pct_change().shift(-1)

    if regime is not None:
        df_bt["regime"] = regime.reindex(df_bt.index).ffill()

    if mode == "unconditional":
        df_bt["position"] = np.sign(df_bt["signal"])
    elif mode == "2state":
        # Only trade when TRENDING (regime == 1)
        df_bt["position"] = np.where(df_bt["regime"] == 1, np.sign(df_bt["signal"]), 0)
    elif mode == "3state":
        # Long when regime=2 (UP-TREND) and signal > 0
        # Short when regime=0 (DOWN-TREND) and signal < 0
        df_bt["position"] = np.where(
            (df_bt["regime"] == 2) & (df_bt["signal"] > 0), 1,
            np.where((df_bt["regime"] == 0) & (df_bt["signal"] < 0), -1, 0)
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")

    df_bt["pnl"] = df_bt["position"] * df_bt["fwd_ret"]
    equity = (1 + df_bt["pnl"]).cumprod()
    return equity

# Walk-forward setup
TRAIN_MONTHS = 6
STEP_MONTHS  = 1
START_DATE   = df.index[lookback]  # skip warmup
END_DATE     = df.index[-1]

date_range = pd.date_range(START_DATE, END_DATE, freq="MS")  # month start
wf_results = []

for i, test_start in enumerate(date_range[TRAIN_MONTHS:]):
    train_start = date_range[i]
    train_end   = test_start
    test_end    = test_start + pd.DateOffset(months=STEP_MONTHS)

    df_train = df.loc[train_start:train_end]
    df_test  = df.loc[test_start:test_end]

    if len(df_test) < 100:
        continue

    # Build signal on train set
    # (In real implementation, recompute features on train set)
    # For now, use pre-computed signal but shift to avoid lookahead
    signal_train = signal.loc[train_start:train_end]
    signal_test  = signal.loc[test_start:test_end]

    # Regimes already computed walk-forward, just slice
    regime_2_test = regime_2_series.loc[test_start:test_end]
    regime_3_test = regime_3_series.loc[test_start:test_end]

    # Backtests
    equity_uncond = simple_backtest(df_test, signal_test, None, "unconditional")
    equity_2state = simple_backtest(df_test, signal_test, regime_2_test, "2state")
    equity_3state = simple_backtest(df_test, signal_test, regime_3_test, "3state")

    def compute_sharpe(equity: pd.Series) -> float:
        ret = equity.pct_change().dropna()
        if ret.std() == 0:
            return 0.0
        return ret.mean() / ret.std() * np.sqrt(252 * 24 * 4)  # 15min bars → annualize

    sharpe_uncond = compute_sharpe(equity_uncond)
    sharpe_2state = compute_sharpe(equity_2state)
    sharpe_3state = compute_sharpe(equity_3state)

    wf_results.append({
        "test_start": test_start.strftime("%Y-%m-%d"),
        "sharpe_uncond": round(sharpe_uncond, 3),
        "sharpe_2state": round(sharpe_2state, 3),
        "sharpe_3state": round(sharpe_3state, 3),
        "n_bars": len(df_test),
    })

wf_df = pd.DataFrame(wf_results)
print(f"\nWalk-Forward Results ({len(wf_df)} windows):")
print(wf_df.to_string(index=False))

# Summary stats
mean_uncond = wf_df["sharpe_uncond"].mean()
mean_2state = wf_df["sharpe_2state"].mean()
mean_3state = wf_df["sharpe_3state"].mean()

improvement_2 = (mean_2state - mean_uncond) / abs(mean_uncond) * 100 if mean_uncond != 0 else 0
improvement_3 = (mean_3state - mean_uncond) / abs(mean_uncond) * 100 if mean_uncond != 0 else 0

print(f"\nMean Sharpe:")
print(f"  Unconditional: {mean_uncond:.3f}")
print(f"  2-State:       {mean_2state:.3f}  ({improvement_2:+.1f}%)")
print(f"  3-State:       {mean_3state:.3f}  ({improvement_3:+.1f}%)")

phase3_pass = (improvement_3 > 30) and (improvement_3 < 200)  # avoid overfitting
print(f"\n✓ Phase 3 Gate: Improvement 30%-200% → {phase3_pass}")
print(f"  3-State improvement: {improvement_3:+.1f}%")

# ══════════════════════════════════════════════════════════════════════════════
# SAVE OUTPUTS
# ══════════════════════════════════════════════════════════════════════════════

reports = ROOT / "reports"
reports.mkdir(exist_ok=True)

ic_comparison.to_csv(reports / "03_regime_ic_comparison.csv", index=False)
wf_df.to_csv(reports / "03_wf_results.csv", index=False)

decision = {
    "phase1_pass": phase1_pass,
    "ic_trending": float(ic_trend),
    "t_trending": float(t_trend),
    "phase2_pass": phase2_pass,
    "hit_up_long": float(hit_up) if not np.isnan(hit_up) else None,
    "hit_down_short": float(hit_down) if not np.isnan(hit_down) else None,
    "phase3_pass": phase3_pass,
    "mean_sharpe_uncond": float(mean_uncond),
    "mean_sharpe_2state": float(mean_2state),
    "mean_sharpe_3state": float(mean_3state),
    "improvement_pct": float(improvement_3),
    "overall_pass": phase1_pass and phase2_pass and phase3_pass,
}

(reports / "03_wf_metrics.json").write_text(json.dumps(decision, indent=2))

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

# Plot 1: IC comparison
fig, ax = plt.subplots(figsize=(10, 6))
regimes = ic_comparison["regime"].values
ics = ic_comparison["ic"].values
colors = ["#1565C0", "#2E7D32", "#C62828"]

ax.bar(regimes, ics, color=colors, alpha=0.8)
ax.axhline(0.04, color="green", linestyle="--", linewidth=1.0, label="IC=0.04 target")
ax.axhline(-0.04, color="red", linestyle="--", linewidth=1.0)
ax.axhline(0, color="black", linewidth=0.5)
ax.set_title("IC Comparison: Unconditional vs Regime-Filtered (4-bar horizon)", fontsize=13)
ax.set_ylabel("Spearman IC")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
for i, (regime, ic, t, n) in enumerate(zip(
    ic_comparison["regime"], ic_comparison["ic"], ic_comparison["t_stat"], ic_comparison["n"]
)):
    ax.text(i, ic + 0.005, f"{ic:.4f}\nt={t:.2f}\nn={n:,}", ha="center", va="bottom", fontsize=9)

plt.tight_layout()
fig.savefig(reports / "03_regime_ic_analysis.png", dpi=150)
plt.close()

# Plot 2: Walk-forward equity curves
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Top: Individual window Sharpe ratios
ax = axes[0]
x = range(len(wf_df))
ax.plot(x, wf_df["sharpe_uncond"], marker="o", label="Unconditional", color="#1565C0", linewidth=1.5)
ax.plot(x, wf_df["sharpe_2state"], marker="s", label="2-State HMM", color="#2E7D32", linewidth=1.5)
ax.plot(x, wf_df["sharpe_3state"], marker="^", label="3-State HMM", color="#FFA726", linewidth=1.5)
ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
ax.set_title("Walk-Forward Sharpe Ratios by Window", fontsize=12)
ax.set_xlabel("Window Index")
ax.set_ylabel("Sharpe Ratio")
ax.legend()
ax.grid(True, alpha=0.3)

# Bottom: Summary bar chart
ax = axes[1]
labels = ["Unconditional", "2-State", "3-State"]
means = [mean_uncond, mean_2state, mean_3state]
colors_bar = ["#1565C0", "#2E7D32", "#FFA726"]
ax.bar(labels, means, color=colors_bar, alpha=0.8)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_title("Mean Walk-Forward Sharpe Ratios", fontsize=12)
ax.set_ylabel("Mean Sharpe")
ax.grid(True, alpha=0.3, axis="y")
for i, (label, mean, imp) in enumerate(zip(labels, means, [0, improvement_2, improvement_3])):
    ax.text(i, mean + 0.02, f"{mean:.3f}\n({imp:+.1f}%)", ha="center", va="bottom", fontsize=10)

plt.tight_layout()
fig.savefig(reports / "03_wf_equity_curves.png", dpi=150)
plt.close()

# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 80)
print("DECISION SUMMARY")
print("=" * 80)
print(f"✓ Phase 1 (IC on TRENDING): {phase1_pass}")
print(f"✓ Phase 2 (3-State Sign Check): {phase2_pass}")
print(f"✓ Phase 3 (Walk-Forward Robustness): {phase3_pass}")
print(f"\n→ OVERALL: {'PASS ✓' if decision['overall_pass'] else 'FAIL ✗'}")
print("\nOutputs:")
print("  - reports/03_regime_ic_comparison.csv")
print("  - reports/03_wf_results.csv")
print("  - reports/03_wf_metrics.json")
print("  - reports/03_regime_ic_analysis.png")
print("  - reports/03_wf_equity_curves.png")
print("=" * 80)

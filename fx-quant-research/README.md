# fx-quant-research — XAUUSD 15min ALMA Trend Framework

A Python-based quantitative FX research framework for evaluating and validating
the **ALMA Slope Trend hypothesis** on XAUUSD 15-minute bars.

---

## Hypothesis

Translated from `StrategyATrend_AlmaSlope` (C# / cAlgo):

> When the ATR-normalised first-difference of an ALMA(50) moving average has been
> consistently positive (or negative) for ≥ 2 consecutive bars during active
> London / New York sessions, enter long (or short) with ATR-based SL/TP sizing.

---

## Project Structure

```
fx-quant-research/
├── config/config.yaml          # All parameters — single source of truth
├── data/
│   ├── raw/XAUUSD15min.csv     # Raw MT-export data (2019–2026)
│   └── processed/              # Parquet cache (auto-generated)
├── src/
│   ├── data/loader.py          # Load + clean pipeline
│   ├── features/
│   │   ├── trend.py            # ALMA, normalised slope, consecutive bar counter
│   │   ├── volatility.py       # ATR
│   │   └── microstructure.py   # Session mask filter
│   ├── strategies/
│   │   └── hypothesis_strategy.py  # Signal generator (+1/-1/0) — isolated
│   ├── backtest/
│   │   ├── engine.py           # Event-driven backtester
│   │   └── metrics.py          # Sharpe, CAGR, MaxDD, profit factor, etc.
│   ├── risk/
│   │   └── position_sizing.py  # ATR fractional sizing + circuit-breaker
│   ├── portfolio/tracker.py    # Equity curve + trade log
│   └── execution/fills.py      # Spread + slippage simulation
├── notebooks/
│   ├── 01_quick_test.py        # Phase 0: visual signal verification
│   └── 02_univariate_ic_test.py # Phase 1: Spearman IC at multiple horizons
├── experiments/
│   ├── exp_baseline.py         # Default-param full backtest
│   ├── exp_parameter_sweep.py  # Grid search over 243 combinations
│   └── exp_risk_variation.py   # Sharpe vs risk% and SL multiplier heatmap
├── reports/                    # All outputs land here (CSV, PNG, JSON)
└── tests/test_framework.py     # 26 unit + integration tests
```

---

## Quick Start

```bash
# 1. Install dependencies
pip install pandas numpy matplotlib pyyaml pyarrow scipy pytest

# 2. Run the baseline backtest
python experiments/exp_baseline.py

# 3. Visual signal check (Phase 0)
python notebooks/01_quick_test.py

# 4. IC analysis (Phase 1)
python notebooks/02_univariate_ic_test.py

# 5. Run all tests
python -m pytest tests/ -v
```

---

## Baseline Results (Default Parameters)

| Metric | Value |
|---|---|
| Trades | 7,325 |
| Sharpe | 0.29 |
| CAGR | 4.3% |
| Total Return | 32.1% |
| Max Drawdown | −61.5% |
| Win Rate | 37.2% |
| Profit Factor | 1.02 |
| Avg Bars Held | 16.3 |
| Exit: SL / TP / TIME | 3974 / 2009 / 916 |

**Key IC finding:** The raw ALMA slope has a **negative** IC of −0.016 (|t|=6.8),
statistically significant at all horizons. This means the slope is a
**mean-reversion** signal in the raw form, not trend-following. The long-only
hit rate of 50.4–51.6% suggests longs slightly outperform shorts.

**Recommended next steps:**
1. Run `exp_parameter_sweep.py` to find parameter combinations that improve Sharpe
2. Run `exp_risk_variation.py` to size for acceptable drawdown given withdrawal targets
3. Implement Notebook 03 (stationarity) and 04 (cross-pair validation)
4. Consider inverting the signal (mean-reversion entry vs trend-following exit)

---

## Configuration

All parameters live in `config/config.yaml`:

```yaml
strategy:
  alma_length: 50         # ALMA lookback window
  min_slope_atr: 0.05     # ATR-normalised slope threshold
  require_consecutive: 2  # Bars of consistent slope before entry

risk:
  risk_pct: 0.75          # % equity per trade
  sl_atr_mult: 2.0        # Stop = ATR × mult
  tp_atr_mult: 4.0        # TP = ATR × mult (1:2 R:R)
  max_daily_loss_pct: 5.0 # Circuit-breaker

execution:
  spread_pips: 3.0        # Fixed spread
  slippage_pips: 1.0      # Max random slippage
  commission_per_lot: 7.0 # USD round-turn
```

---

## Design Principles

- **Hypothesis isolation** — `hypothesis_strategy.py` contains only signal logic;
  no sizing, no execution details bleed in.
- **Decoupled risk** — `RiskManager` answers "how much?" and "where?"; the
  strategy never knows position sizes.
- **No look-ahead** — signals are shifted +1 bar before the engine sees them.
- **Realistic execution** — spread + slippage on every fill, commission on
  round-turns, SL/TP triggered at the price level (not at close).
- **Config-driven** — swap parameters without touching code.

---

## Pipeline Alignment (19-Notebook Spec)

| Notebook | Script | Status |
|---|---|---|
| 01 Quick Test | `notebooks/01_quick_test.py` | ✅ |
| 02 IC Test | `notebooks/02_univariate_ic_test.py` | ✅ |
| 06 Initial Backtest | `experiments/exp_baseline.py` | ✅ |
| 09 Parameter Stability | `experiments/exp_parameter_sweep.py` | ✅ |
| 11 Cost Sensitivity | `experiments/exp_risk_variation.py` | ✅ |
| 03–05, 07–08, 10, 12–19 | — | Pending |

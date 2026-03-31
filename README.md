# FX Quant Research — XAUUSD 15min ALMA Trend Framework

This repository contains a Python-based quantitative FX research framework for evaluating and validating the **ALMA Slope Trend hypothesis** on XAUUSD 15-minute bars.

## 📂 Repository Structure

This is a container repository. The main project is located in the `fx-quant-research/` subdirectory.

```
fx-quant-research--1-/
└── fx-quant-research/          # Main research framework
    ├── README.md               # Full project documentation
    ├── config/                 # Strategy parameters (YAML)
    ├── src/                    # Python modules
    ├── experiments/            # Backtest scripts
    ├── notebooks/              # Analysis notebooks
    ├── reports/                # Generated results
    └── tests/                  # Unit tests
```

## 🎯 Hypothesis

**ATR-normalized ALMA slope trend detection:**

> When the ATR-normalized first-difference of an ALMA(50) moving average has been consistently positive (or negative) for ≥ 2 consecutive bars during active London/New York sessions, enter long (or short) with ATR-based SL/TP sizing.

## 📊 Key Results (Baseline)

| Metric | Value |
|--------|-------|
| Trades | 7,325 |
| Sharpe Ratio | 0.29 |
| CAGR | 4.3% |
| Max Drawdown | −61.5% |
| Win Rate | 37.2% |
| Profit Factor | 1.02 |

**IC Finding:** Raw ALMA slope has negative IC of −0.016, suggesting mean-reversion characteristics rather than trend-following.

## 🚀 Quick Start

Navigate to the main project directory:

```bash
cd fx-quant-research
```

Then follow the instructions in [`fx-quant-research/README.md`](fx-quant-research/README.md).

### Installation

```bash
pip install pandas numpy matplotlib pyyaml pyarrow scipy pytest
```

### Run Baseline Backtest

```bash
python experiments/exp_baseline.py
```

### Visual Signal Verification

```bash
python notebooks/01_quick_test.py
```

### IC Analysis

```bash
python notebooks/02_univariate_ic_test.py
```

## 🏗️ Framework Features

- **Hypothesis Isolation** — Clean separation between signal logic and execution
- **No Look-Ahead Bias** — Signals shifted +1 bar before engine processes
- **Realistic Execution** — Spread, slippage, commission on every trade
- **Config-Driven** — All parameters in `config/config.yaml`
- **Comprehensive Testing** — 26 unit + integration tests

## 📈 Research Pipeline

1. ✅ **Quick Test** — Visual signal verification
2. ✅ **IC Test** — Spearman correlation analysis
3. ✅ **Baseline Backtest** — Default parameter performance
4. ✅ **Parameter Sweep** — Grid search over 243 combinations
5. ✅ **Risk Variation** — Sharpe vs risk% heatmap
6. ⏳ **Cross-Pair Validation** — Test on other FX pairs
7. ⏳ **Regime Detection** — Volatility-conditional performance
8. ⏳ **Production Deployment** — MT5 integration

## 🛠️ Tech Stack

- **Python 3.8+** — Core language
- **Pandas** — Data manipulation
- **NumPy** — Numerical computation
- **PyYAML** — Configuration management
- **Matplotlib** — Visualization
- **pytest** — Testing framework

## 📚 Documentation

Full documentation is available in the main project directory:

- [Main README](fx-quant-research/README.md) — Complete project documentation
- [Configuration Guide](fx-quant-research/config/config.yaml) — Parameter reference
- [Test Results](fx-quant-research/reports/) — Generated analysis outputs

## 🔬 Experimental Scripts

Located in `fx-quant-research/experiments/`:

- `exp_baseline.py` — Default parameter backtest
- `exp_parameter_sweep.py` — Grid search (243 combinations)
- `exp_risk_variation.py` — Risk sizing optimization

## ⚠️ Research Status

**Current Phase:** Parameter optimization and IC validation

**Next Steps:**
1. Expand parameter sweep to improve Sharpe ratio
2. Implement cross-pair validation (EURUSD, GBPUSD)
3. Add regime-conditional analysis
4. Consider signal inversion (mean-reversion vs trend-following)

## 📝 License

MIT License (or specify your license)

## 📧 Contact

For questions: [Your GitHub Profile](https://github.com/FredrickMbithi)

---

**Asset:** XAUUSD (Gold)  
**Timeframe:** 15-minute bars  
**Data Range:** 2019–2026  
**Framework:** Python + Pandas + NumPy

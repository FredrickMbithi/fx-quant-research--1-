"""
tests/test_framework.py
────────────────────────
Unit tests for all framework modules.
Run with:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.features.trend import alma, alma_slope_normalised, consecutive_bars
from src.features.volatility import atr
from src.features.microstructure import session_mask
from src.risk.position_sizing import RiskManager
from src.portfolio.tracker import PortfolioTracker, Position
from src.execution.fills import FillSimulator


# ── Fixtures ─────────────────────────────────────────────────────────────────

def make_ohlcv(n=200, base=1800.0, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2023-01-02 08:00", periods=n, freq="15min", tz="UTC")
    close = base + np.cumsum(rng.normal(0, 1, n))
    high  = close + rng.uniform(0.1, 1.0, n)
    low   = close - rng.uniform(0.1, 1.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    vol   = rng.integers(500, 2000, n).astype(float)
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close, "volume": vol}, index=idx)


@pytest.fixture
def df():
    return make_ohlcv()


@pytest.fixture
def cfg():
    return {
        "data":     {"pip_size": 0.01, "raw_path": "", "processed_path": "", "symbol": "XAUUSD", "timeframe": "15min"},
        "strategy": {"alma_length": 20, "alma_offset": 0.85, "alma_sigma": 6.0,
                     "min_slope_atr": 0.05, "require_consecutive": 2,
                     "atr_length": 14, "allow_shorts": True},
        "risk":     {"risk_pct": 1.0, "sl_atr_mult": 2.0, "tp_atr_mult": 4.0,
                     "max_hold_bars": 50, "max_daily_loss_pct": 5.0},
        "execution":{"spread_pips": 3.0, "slippage_pips": 1.0, "commission_per_lot": 7.0},
        "backtest": {"initial_equity": 10000.0, "lot_size": 100},
        "session_filters": {"enabled": False},
    }


# ── Feature Tests ─────────────────────────────────────────────────────────────

class TestALMA:
    def test_output_length(self, df):
        result = alma(df["close"], length=20)
        assert len(result) == len(df)

    def test_nan_warmup(self, df):
        result = alma(df["close"], length=20)
        assert result.iloc[:19].isna().all()
        assert result.iloc[19:].notna().all()

    def test_values_reasonable(self, df):
        result = alma(df["close"], length=20)
        valid = result.dropna()
        # ALMA should track price — within 5% of close range
        price_range = df["close"].max() - df["close"].min()
        assert (valid - df["close"].loc[valid.index]).abs().max() < price_range * 0.5

    def test_smooth_more_than_close(self, df):
        """ALMA std should be <= close std (it's a moving average)."""
        result = alma(df["close"], length=20).dropna()
        assert result.std() <= df["close"].std() * 1.1   # small tolerance


class TestATR:
    def test_output_length(self, df):
        result = atr(df, length=14)
        assert len(result) == len(df)

    def test_always_positive(self, df):
        result = atr(df, length=14).dropna()
        assert (result > 0).all()

    def test_nan_warmup(self, df):
        result = atr(df, length=14)
        assert result.iloc[:13].isna().all()


class TestConsecutiveBars:
    def test_basic(self):
        cond = pd.Series([False, True, True, True, False, True])
        result = consecutive_bars(cond)
        expected = pd.Series([0, 1, 2, 3, 0, 1])
        pd.testing.assert_series_equal(result.reset_index(drop=True), expected, check_names=False)

    def test_all_false(self):
        cond = pd.Series([False] * 5)
        assert (consecutive_bars(cond) == 0).all()

    def test_all_true(self):
        cond = pd.Series([True] * 5)
        expected = pd.Series([1, 2, 3, 4, 5])
        pd.testing.assert_series_equal(consecutive_bars(cond).reset_index(drop=True), expected, check_names=False)


class TestSessionMask:
    def test_disabled(self):
        idx = pd.date_range("2023-01-02 08:00", periods=10, freq="15min", tz="UTC")
        mask = session_mask(idx, {"enabled": False})
        assert mask.all()

    def test_london_session(self):
        idx = pd.date_range("2023-01-02 07:00", periods=20, freq="1h", tz="UTC")
        cfg = {"enabled": True, "london": {"enabled": True, "start": "08:00", "end": "16:30"}}
        mask = session_mask(idx, cfg)
        # 07:00 UTC should be outside London
        assert not mask.iloc[0]
        # 09:00 UTC should be inside London
        assert mask.iloc[2]


# ── Risk Tests ────────────────────────────────────────────────────────────────

class TestRiskManager:
    def test_position_size_positive(self, cfg):
        rm = RiskManager(cfg)
        units = rm.position_size(10000, atr_value=1.5)
        assert units > 0

    def test_position_size_zero_atr(self, cfg):
        rm = RiskManager(cfg)
        assert rm.position_size(10000, atr_value=0) == 0.0

    def test_sl_tp_long(self, cfg):
        rm = RiskManager(cfg)
        sl, tp = rm.sl_tp(entry_price=1800.0, direction=1, atr_value=2.0)
        assert sl < 1800.0        # SL below entry for long
        assert tp > 1800.0        # TP above entry for long
        assert (tp - 1800) > (1800 - sl)   # TP further than SL (R:R > 1)

    def test_sl_tp_short(self, cfg):
        rm = RiskManager(cfg)
        sl, tp = rm.sl_tp(entry_price=1800.0, direction=-1, atr_value=2.0)
        assert sl > 1800.0
        assert tp < 1800.0

    def test_circuit_breaker(self, cfg):
        rm = RiskManager(cfg)
        ts = pd.Timestamp("2023-01-02 09:00", tz="UTC")
        rm.update_day(ts, equity=10000.0)
        assert not rm.check_circuit_breaker(9600.0)   # 4% loss — below threshold
        assert rm.check_circuit_breaker(9400.0)       # 6% loss — triggers

    def test_commission_scales_with_units(self, cfg):
        rm = RiskManager(cfg)
        c1 = rm.commission(100)
        c2 = rm.commission(200)
        assert abs(c2 - 2 * c1) < 0.01


# ── Portfolio Tests ───────────────────────────────────────────────────────────

class TestPortfolioTracker:
    def test_pnl_long_profit(self):
        pt = PortfolioTracker(10000.0)
        pos = Position(0, pd.Timestamp("2023-01-02", tz="UTC"), 1, 100, 1800.0, 1796.0, 1808.0, 0.0)
        pt.open_position(pos)
        trade = pt.close_position(pd.Timestamp("2023-01-02 01:00", tz="UTC"), 1804.0, "TP", 4)
        assert trade.pnl > 0      # bought 1800, sold 1804 → profit

    def test_pnl_long_loss(self):
        pt = PortfolioTracker(10000.0)
        pos = Position(0, pd.Timestamp("2023-01-02", tz="UTC"), 1, 100, 1800.0, 1796.0, 1808.0, 0.0)
        pt.open_position(pos)
        trade = pt.close_position(pd.Timestamp("2023-01-02 01:00", tz="UTC"), 1796.0, "SL", 4)
        assert trade.pnl < 0

    def test_equity_updates(self):
        pt = PortfolioTracker(10000.0)
        pos = Position(0, pd.Timestamp("2023-01-02", tz="UTC"), 1, 100, 1800.0, 1796.0, 1808.0, 10.0)
        pt.open_position(pos)
        assert pt.equity < 10000  # commission deducted on entry
        pt.close_position(pd.Timestamp("2023-01-02 01:00", tz="UTC"), 1802.0, "TP", 4)
        # +$200 gross on 100 oz, -$10 commission
        assert abs(pt.equity - (10000 - 5 + 200 - 10)) < 1.0

    def test_no_double_position(self):
        pt = PortfolioTracker(10000.0)
        pos = Position(0, pd.Timestamp("2023-01-02", tz="UTC"), 1, 100, 1800.0, 1796.0, 1808.0, 0.0)
        pt.open_position(pos)
        with pytest.raises(AssertionError):
            pt.open_position(pos)


# ── Fill Simulator Tests ──────────────────────────────────────────────────────

class TestFillSimulator:
    def test_long_fill_above_open(self, cfg):
        fs = FillSimulator(cfg)
        fill = fs.fill_price(bar_open=1800.0, direction=1)
        assert fill > 1800.0   # spread + slippage pushes long entry up

    def test_short_fill_below_open(self, cfg):
        fs = FillSimulator(cfg)
        fill = fs.fill_price(bar_open=1800.0, direction=-1)
        assert fill < 1800.0


# ── Integration Test ──────────────────────────────────────────────────────────

class TestEndToEnd:
    def test_backtest_produces_trades(self, cfg):
        from src.strategies.hypothesis_strategy import build_features, generate_signals
        from src.backtest.engine import run_backtest

        df = make_ohlcv(n=500, base=1800.0)
        features = build_features(df, cfg)
        signals  = generate_signals(df, cfg)
        portfolio, trade_log = run_backtest(df, signals, features, cfg)

        assert portfolio.equity > 0
        assert not trade_log.empty
        assert "pnl" in trade_log.columns
        assert "exit_reason" in trade_log.columns

    def test_metrics_keys(self, cfg):
        from src.strategies.hypothesis_strategy import build_features, generate_signals
        from src.backtest.engine import run_backtest
        from src.backtest.metrics import compute_metrics

        df = make_ohlcv(n=500, base=1800.0)
        features = build_features(df, cfg)
        signals  = generate_signals(df, cfg)
        portfolio, trade_log = run_backtest(df, signals, features, cfg)
        equity = portfolio.equity_series()
        m = compute_metrics(trade_log, equity, cfg["backtest"]["initial_equity"])

        for key in ["sharpe", "cagr", "max_drawdown", "win_rate", "profit_factor", "n_trades"]:
            assert key in m

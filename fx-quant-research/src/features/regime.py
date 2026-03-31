"""
src/features/regime.py
───────────────────────
Hidden Markov Model (HMM) regime filter for XAUUSD.

Fits a 2-state Gaussian HMM on return volatility features and labels each bar
as TRENDING (state associated with lower volatility / directional moves) or
RANGING (state associated with higher volatility / chop).

Design principles:
  • Trained only on data BEFORE the bar being labelled (walk-forward safe).
  • States are identified post-hoc by their mean absolute return:
      lower mean |ret| → TRENDING (directional, smoother)
      higher mean |ret| → RANGING (choppy, mean-reverting)
  • Returns integer series: 1 = TRENDING, 0 = RANGING.

Usage
-----
    from src.features.regime import HMMRegimeFilter
    hmm = HMMRegimeFilter(n_states=2, lookback=2000)
    regimes = hmm.fit_predict(df)
"""
from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

TRENDING = 1
RANGING  = 0


class HMMRegimeFilter:
    """
    Walk-forward HMM regime labeller.

    Parameters
    ----------
    n_states    : Number of hidden states (2 recommended: trend vs range).
    lookback    : Bars of history used to train each HMM fit.
    refit_every : Refit the model every N bars (balances accuracy vs speed).
    n_iter      : HMM EM iterations per fit.
    random_state: Reproducibility seed.
    """

    def __init__(
        self,
        n_states: int = 2,
        lookback: int = 2000,
        refit_every: int = 500,
        n_iter: int = 100,
        random_state: int = 42,
    ) -> None:
        self.n_states    = n_states
        self.lookback    = lookback
        self.refit_every = refit_every
        self.n_iter      = n_iter
        self.random_state = random_state

    # ------------------------------------------------------------------

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Observation matrix fed to the HMM.
        Features (all normalised):
          1. Log return
          2. Rolling 20-bar realised volatility (std of log-returns)
          3. |log return| / ATR  (normalised range)
          4. Bar range: (high - low) / close
        """
        close  = df["close"].values.astype(float)
        high   = df["high"].values.astype(float)
        low    = df["low"].values.astype(float)

        log_ret  = np.diff(np.log(close), prepend=np.log(close[0]))
        roll_vol = pd.Series(log_ret).rolling(20, min_periods=5).std().fillna(0).values
        bar_range = (high - low) / close

        X = np.column_stack([log_ret, roll_vol, bar_range])
        return X

    def _fit_model(self, X_train: np.ndarray) -> GaussianHMM:
        scaler  = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = GaussianHMM(
                n_components=self.n_states,
                covariance_type="full",
                n_iter=self.n_iter,
                random_state=self.random_state,
                tol=1e-4,
            )
            model.fit(X_scaled)

        model._scaler = scaler   # store for prediction
        return model

    def _label_states(self, model: GaussianHMM, X_train: np.ndarray) -> dict[int, int]:
        """
        Map HMM state ids → TRENDING / RANGING.
        State with lower mean |log_ret| is TRENDING (smoother directional moves).
        """
        scaler   = model._scaler
        X_scaled = scaler.transform(X_train)
        states   = model.predict(X_scaled)

        # Mean absolute log-return per state
        log_ret_col = X_train[:, 0]
        mean_abs = {s: np.mean(np.abs(log_ret_col[states == s]))
                    for s in range(self.n_states)}

        sorted_states = sorted(mean_abs, key=mean_abs.get)
        # sorted_states[0] = lowest mean |ret| = TRENDING
        return {sorted_states[0]: TRENDING, sorted_states[1]: RANGING}

    # ------------------------------------------------------------------

    def fit_predict(self, df: pd.DataFrame) -> pd.Series:
        """
        Walk-forward HMM labelling.

        Returns
        -------
        pd.Series[int]: 1 = TRENDING, 0 = RANGING, same index as df.
        NaN for warmup bars where we don't have enough history.
        """
        X      = self._build_features(df)
        n      = len(df)
        labels = np.full(n, np.nan)

        model      = None
        state_map  = None
        last_fit   = -1

        for i in range(self.lookback, n):
            # Refit when due
            if model is None or (i - last_fit) >= self.refit_every:
                start    = max(0, i - self.lookback)
                X_train  = X[start:i]
                try:
                    model     = self._fit_model(X_train)
                    state_map = self._label_states(model, X_train)
                    last_fit  = i
                except Exception as e:
                    log.warning("HMM fit failed at bar %d: %s", i, e)
                    continue

            # Predict current bar
            scaler  = model._scaler
            x_i     = scaler.transform(X[i:i+1])
            raw_state = model.predict(x_i)[0]
            labels[i] = state_map.get(raw_state, RANGING)

        result = pd.Series(labels, index=df.index, name="regime")
        trending = (result == TRENDING).sum()
        ranging  = (result == RANGING).sum()
        log.info("HMM regimes — TRENDING: %d bars (%.1f%%), RANGING: %d bars (%.1f%%)",
                 trending, 100*trending/n, ranging, 100*ranging/n)
        return result

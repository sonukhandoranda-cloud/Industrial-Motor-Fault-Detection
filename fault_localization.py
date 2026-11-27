"""
Feature 2 - which SIGNAL looks most like the fault?

Your datasets have ONE fault class (Conditions 0 / 1), so this ranks signals
(I_Sw1, V_Sw1, StatorPh Alpha/Beta); it cannot name different fault types.

  deviation       = mean |z| of a signal's 10 features vs the healthy baseline,
                    divided by the typical (median) healthy value  (1.0 = normal)
  fault-likeness  = (deviation - healthy mean) / (training-fault level - healthy mean)
                    0 % = looks healthy, 100 % = as abnormal as your training faults
A signal is called abnormal at >= 50 %.  A signal in which your training faults hardly
differ from healthy data is left out (it cannot tell anything about the fault), and the
two *_freq_axis columns are ignored.
"""
import numpy as np
import pandas as pd

from monitor_common import (MEASURED, SIGNAL_NAMES, STATS, channel_of_feature, feature_label)

ABNORMAL_AT = 0.5
MIN_EXCESS = 0.02
MIN_SEPARATION = 0.5
N_SIG, N_STAT = len(SIGNAL_NAMES), len(STATS)


def _channel_scores(Xs, mu, sd):
    z = np.clip(np.abs((np.asarray(Xs, float) - mu) / sd), 0, 10)
    return np.column_stack([z[:, c * N_STAT:(c + 1) * N_STAT].mean(1) for c in range(N_SIG)])


class FaultLocaliser:
    def __init__(self, stats: dict, names):
        self.s = stats
        self.names = list(names)
        self.mu, self.sd = stats["mu"], stats["sd"]
        self.informative = np.asarray(stats["informative"])
        self._measured_mask = np.array([channel_of_feature(n) in MEASURED for n in self.names])

    @staticmethod
    def fit_stats(X, y, names=None) -> dict:
        """Computed once from YOUR X_train_features / y_train (done by make_baseline.py)."""
        X, y = np.asarray(X, float), np.asarray(y)
        healthy, faulty = X[y == 0], X[y == 1]
        mu, sd = healthy.mean(0), healthy.std(0) + 1e-9
        h_raw = _channel_scores(healthy, mu, sd)
        norm = np.median(h_raw, axis=0) + 1e-9
        h_scores = h_raw / norm
        h_mean = h_scores.mean(0)
        fault_level = (_channel_scores(faulty, mu, sd) / norm).mean(0)
        separation = (fault_level - h_mean) / (h_scores.std(0) + 1e-9)
        informative = separation >= MIN_SEPARATION
        if not informative[MEASURED].any():
            informative[:] = True
        return {"mu": mu, "sd": sd, "norm": norm, "h_mean": h_mean, "fault_level": fault_level,
                "excess": np.maximum(fault_level - h_mean, MIN_EXCESS),
                "separation": separation, "informative": informative}

    def deviation(self, Xs) -> pd.DataFrame:
        s = _channel_scores(Xs, self.mu, self.sd) / self.s["norm"]
        return pd.DataFrame(s[:, MEASURED], columns=[SIGNAL_NAMES[i] for i in MEASURED])

    def fault_likeness(self, Xs) -> pd.DataFrame:
        dev = self.deviation(Xs)
        like = (dev - self.s["h_mean"][MEASURED]) / self.s["excess"][MEASURED]
        like.loc[:, ~self.informative[MEASURED]] = np.nan
        return like

    def localise(self, Xs, rows=None) -> dict:
        like, dev = self.fault_likeness(Xs), self.deviation(Xs)
        if rows is not None and len(rows):
            like, dev = like.iloc[rows], dev.iloc[rows]
        ranking = like.mean().dropna().sort_values(ascending=False)
        return {"ranking": ranking,
                "ignored_signals": [SIGNAL_NAMES[i] for i in MEASURED if not self.informative[i]],
                "deviation": dev.mean().reindex(ranking.index),
                "suspect": ranking.index[0], "suspect_score": float(ranking.iloc[0]),
                "abnormal_signals": ranking.index[ranking >= ABNORMAL_AT].tolist(),
                "abnormal": bool(ranking.iloc[0] >= ABNORMAL_AT)}

    def top_features(self, Xs_row, k=8) -> pd.DataFrame:
        z = np.clip(np.abs((np.asarray(Xs_row, float).reshape(-1) - self.mu) / self.sd), 0, 10)
        z = np.where(self._measured_mask, z, -1)
        idx = np.argsort(z)[::-1][:k]
        return pd.DataFrame({"feature": [feature_label(self.names[i]) for i in idx],
                             "z_vs_healthy": z[idx]})

    def per_window_suspect(self, Xs) -> np.ndarray:
        return self.fault_likeness(Xs).idxmax(axis=1).to_numpy()

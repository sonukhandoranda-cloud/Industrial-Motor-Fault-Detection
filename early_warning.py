"""
Feature 1 - EARLY WARNING that a fault is about to occur.

Instead of only asking "is it faulty NOW?", this tracks how the model's fault
probability and the drift away from the healthy baseline are TRENDING, and warns
when they are rising towards the fault level.

  NORMAL   nothing unusual
  WARNING  P(fault) >= warn_p, OR P(fault) is rising and projected to reach fault_p
           within `horizon_windows`, OR drift from healthy > drift_warn
  FAULT    smoothed P(fault) >= fault_p

Uses your saved fault_model_best.joblib + fault_scaler.joblib and the healthy baseline
in monitor_baseline.joblib.  Only meaningful for a TIME-ORDERED recording; tune the
thresholds on a real recording in which a fault develops.
"""
import numpy as np
import pandas as pd

from monitor_common import (STEP_SIZE, WINDOW_SIZE, drift_score, extract_features,
                            load_baseline, load_model_and_scaler)


class EarlyWarning:
    def __init__(self, model=None, scaler=None, baseline=None, ema=0.2, slope_windows=30,
                 warn_p=0.30, fault_p=0.50, horizon_windows=60, min_slope=0.002,
                 drift_warn=1.5):
        if model is None or scaler is None:
            model, scaler = load_model_and_scaler()
        self.model, self.scaler = model, scaler
        b = baseline or load_baseline()
        self.mu, self.sd, self.drift_ref = b["healthy_mu"], b["healthy_sd"], b["drift_ref"]
        self.ema, self.K = ema, slope_windows
        self.warn_p, self.fault_p = warn_p, fault_p
        self.horizon, self.min_slope, self.drift_warn = horizon_windows, min_slope, drift_warn
        self.last_Xs = None

    def analyse(self, raw: np.ndarray) -> pd.DataFrame:
        """raw (n_samples, 6) -> one row per window: probability, trend, drift, ETA, state."""
        Xs = np.nan_to_num(self.scaler.transform(extract_features(raw)))   # constant window -> NaN -> 0
        self.last_Xs = Xs                                   # reused for SHAP / localisation
        p = self.model.predict_proba(Xs)[:, 1]
        drift = drift_score(Xs, self.mu, self.sd) / self.drift_ref      # 1.0 = edge of healthy

        p_s = pd.Series(p).ewm(alpha=self.ema, adjust=False).mean().values
        slope = np.zeros_like(p_s)
        for i in range(len(p_s)):
            lo = max(0, i - self.K + 1)
            if i - lo >= 4:
                slope[i] = np.polyfit(np.arange(lo, i + 1), p_s[lo:i + 1], 1)[0]

        eta = np.full(len(p_s), np.inf)
        rising = slope > self.min_slope
        eta[rising] = np.maximum(0, (self.fault_p - p_s[rising]) / slope[rising])

        state = np.where(
            p_s >= self.fault_p, "FAULT",
            np.where((p_s >= self.warn_p) | (rising & (eta <= self.horizon)) |
                     (drift > self.drift_warn), "WARNING", "NORMAL"))
        return pd.DataFrame({
            "window": np.arange(len(p)),
            "sample_index": np.arange(len(p)) * STEP_SIZE + WINDOW_SIZE - 1,
            "p_fault": p, "p_smooth": p_s, "slope": slope, "drift": drift,
            "eta_windows": eta, "eta_samples": eta * STEP_SIZE, "state": state})

    def first_warning(self, result: pd.DataFrame):
        w = result[result.state != "NORMAL"]
        return None if w.empty else w.iloc[0]


def simulate_degradation(healthy_raw, faulty_raw, n=60000, seed=0):
    """DEMO ONLY: blends YOUR healthy data into YOUR faulty data with a rising faulty
    fraction, because you have no recording in which a fault develops slowly."""
    rng = np.random.default_rng(seed)
    n = min(n, len(healthy_raw))
    out = []
    for i in range(0, n, WINDOW_SIZE):
        src = faulty_raw if rng.random() < (i / n) ** 2 else healthy_raw
        j = rng.integers(0, len(src) - WINDOW_SIZE)
        out.append(src[j:j + WINDOW_SIZE])
    out = np.vstack(out)
    # a plain increasing x-axis column (your frequency axes) must stay one continuous ramp,
    # otherwise every stitched block joint looks like a huge jump
    for c in range(out.shape[1]):
        d = np.diff(healthy_raw[:, c])
        if len(d) and np.all(d > 0):
            out[:, c] = healthy_raw[0, c] + np.median(d) * np.arange(len(out))
    return out

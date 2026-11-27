"""
Shared helpers for the monitoring add-ons (early warning, fault localisation,
SHAP explanation, reports).  Nothing in your notebooks or app.py is changed.

Needs next to these files (already in your repo):
    fault_model_best.joblib  fault_scaler.joblib  feature_names.joblib
and ONE extra file made from YOUR data (run `python make_baseline.py` once):
    monitor_baseline.joblib
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from scipy.stats import kurtosis, skew

BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "fault_model_best.joblib"
SCALER_FILE = BASE_DIR / "fault_scaler.joblib"
NAMES_FILE = BASE_DIR / "feature_names.joblib"
BASELINE_FILE = BASE_DIR / "monitor_baseline.joblib"
# only make_baseline.py reads these two (they are git-ignored in your repo)
XTRAIN_FILE = BASE_DIR / "X_train_features.csv"
YTRAIN_FILE = BASE_DIR / "y_train.npy"

WINDOW_SIZE, STEP_SIZE = 20, 5            # same as your feature-engineering notebook

# Columns of your datasets, in the order the notebooks call Feature_1 .. Feature_6
SIGNAL_NAMES = ["I_Sw1", "I_Sw1_freq_axis", "V_Sw1", "V_Sw1_freq_axis",
                "StatorPh_CurrentAlpha", "StatorPh_CurrentBeta"]
MEASURED = [0, 2, 4, 5]                    # the two *_freq_axis columns are x-axes
STATS = ["mean", "std", "rms", "min", "max", "ptp", "skew", "kurt", "dom_freq", "dom_mag"]
STAT_LABEL = {"ptp": "peak-to-peak", "kurt": "kurtosis", "dom_freq": "dominant freq",
              "dom_mag": "dominant magnitude"}
# feature columns (0-59) that belong to the 4 measured signals
MEAS_COLS = np.r_[0:10, 20:30, 40:60]


def feature_label(name: str) -> str:
    """'Feature_5_kurt' -> 'StatorPh_CurrentAlpha · kurtosis'"""
    try:
        _, idx, *stat = name.split("_")
        stat = "_".join(stat)
        return f"{SIGNAL_NAMES[int(idx) - 1]} · {STAT_LABEL.get(stat, stat)}"
    except Exception:
        return name


def channel_of_feature(name: str) -> int:
    return int(name.split("_")[1]) - 1


def extract_features(raw: np.ndarray, window_size=WINDOW_SIZE, step_size=STEP_SIZE) -> np.ndarray:
    """Same 10 features per signal as your notebook (checked identical), but vectorised:
    a 500k-row recording takes seconds instead of many minutes.
    raw: (n_samples, 6) in the order of your dataset columns."""
    # Column-contiguous (Fortran) layout, exactly what your notebook's DataFrame.values has.
    # It matters: Feature_2_skew / Feature_4_skew come from the straight frequency ramp, are
    # pure rounding noise and your scaler divides them by ~2e-11, so a different memory
    # layout (different summation order) would change the model input.
    raw = np.asfortranarray(raw, dtype=float)
    w = sliding_window_view(raw, window_size, axis=0)[::step_size]   # (windows, signals, window)
    mn, mx = w.min(-1), w.max(-1)
    mag = np.abs(np.fft.rfft(w, axis=-1))
    d = np.argmax(mag[..., 1:], axis=-1) + 1                          # skip the DC bin
    dom_f = np.fft.rfftfreq(window_size)[d]
    dom_m = np.take_along_axis(mag, d[..., None], -1)[..., 0]
    f = np.stack([w.mean(-1), w.std(-1), np.sqrt((w ** 2).mean(-1)), mn, mx, mx - mn,
                  skew(w, axis=-1), kurtosis(w, axis=-1), dom_f, dom_m], axis=-1)
    return f.reshape(len(f), -1)


def drift_score(Xs, mu, sd):
    """Mean |z| of the measured-signal features vs the healthy baseline (z clipped at 10).
    The *_freq_axis columns are ignored: their window statistics are floating-point noise
    (your scaler shows Feature_2_skew with a scale of ~2e-11)."""
    z = np.abs((np.asarray(Xs, float) - mu) / sd)[:, MEAS_COLS]
    return np.mean(np.clip(z, 0, 10), axis=1)


def load_feature_names():
    return list(joblib.load(NAMES_FILE))


def load_model_and_scaler():
    return joblib.load(MODEL_FILE), joblib.load(SCALER_FILE)


def load_baseline() -> dict:
    if not BASELINE_FILE.exists():
        raise FileNotFoundError(
            "monitor_baseline.joblib not found. Run `python make_baseline.py` once in the "
            "folder that contains X_train_features.csv and y_train.npy.")
    return joblib.load(BASELINE_FILE)


def read_signal_file(file, max_rows=None, return_labels=False):
    """Read an uploaded .csv/.xlsx and return the first 6 numeric columns as (n, 6).
    With return_labels=True also returns the 7th column (Condition: 0 healthy / 1 faulty)
    if the file has one, otherwise None."""
    name = getattr(file, "name", str(file)).lower()
    df = pd.read_excel(file, nrows=max_rows) if name.endswith((".xlsx", ".xls")) \
        else pd.read_csv(file, nrows=max_rows)
    df = df.select_dtypes(include=[np.number])
    if df.shape[1] < 6:
        raise ValueError(f"Need at least 6 numeric columns (found {df.shape[1]}), in the "
                         "same order as your training datasets.")
    raw = df.iloc[:, :6].to_numpy(dtype=float)
    if not return_labels:
        return raw
    labels = None
    if df.shape[1] >= 7:
        col = df.iloc[:, 6].to_numpy()
        if np.isin(np.unique(col[~np.isnan(col)]), [0, 1]).all():
            labels = col.astype(int)
    return raw, labels


def fault_start_from(labels=None, healthy_rows=0):
    """Index (0-based) of the first faulty row, or None if unknown / no faulty rows.
    `healthy_rows` typed by the user wins over the Condition column."""
    if healthy_rows and healthy_rows > 0:
        return int(healthy_rows)
    if labels is not None:
        ones = np.where(np.asarray(labels) == 1)[0]
        return int(ones[0]) if len(ones) else None
    return None

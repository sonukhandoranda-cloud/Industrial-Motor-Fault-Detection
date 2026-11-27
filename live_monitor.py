"""
REAL-TIME early warning.  Feed new samples as they arrive; get warnings while a fault is
still developing.

    mon = StreamMonitor()                       # loads your model, scaler, baseline
    new_windows, new_alerts = mon.push(rows)    # rows: (k, 6) array in dataset column order
    mon.level_name                              # 'NORMAL' / 'WARNING' / 'FAULT'

Same rules as early_warning.py (same numbers on the same data), made incremental:
  * windows of 20 samples, step 5, built as data arrives (nothing is re-computed)
  * WARNING  = P(fault) high, OR rising and projected to reach the fault level soon,
               OR drift from healthy above `drift_warn` (set None to switch drift off)
  * FAULT    = smoothed P(fault) >= fault_p
  * to avoid false alarms a level is only raised after it persisted for
    `confirm_warning` / `confirm_fault` windows in a row, and only cleared after
    `clear_after` calm windows.

Command line (watches a CSV that your logger / PLC script keeps appending rows to):
    python live_monitor.py --file live_data.csv
"""
import argparse
import re
import time
from collections import deque
from datetime import datetime

import numpy as np
import pandas as pd

from monitor_common import (SIGNAL_NAMES, STEP_SIZE, WINDOW_SIZE, drift_score, extract_features,
                            load_baseline, load_model_and_scaler)

NAMES = ["NORMAL", "WARNING", "FAULT"]

# ------------------------------------------------------------------- sound
# WARNING = two medium beeps, FAULT = three longer, higher beeps, TEST = both.
BEEPS = {"WARNING": [(880, 0.18)] * 2, "FAULT": [(1320, 0.30)] * 3}


def alert_sound_wav(level: str = "WARNING", salt: int = 0) -> bytes:
    """A small WAV (bytes) for the browser. `salt` makes every file slightly different so the
    browser plays it again even when the same level fires twice in a row."""
    import io
    import wave
    rate, parts = 22050, []

    def tone(freq, dur):
        t = np.arange(int(rate * dur)) / rate
        fade = np.minimum(1.0, np.minimum(t, dur - t) / 0.01)          # no clicks
        return 0.6 * np.sin(2 * np.pi * freq * t) * fade

    def pattern(lv):
        out = []
        for f, d in BEEPS[lv]:
            out += [tone(f, d), np.zeros(int(rate * 0.10))]
        return out

    if level == "TEST":
        parts = pattern("WARNING") + [np.zeros(int(rate * 0.5))] + pattern("FAULT")
    else:
        parts = pattern(level if level in BEEPS else "WARNING")
    parts.append(np.zeros(1 + (salt * 7) % 4000))
    pcm = (np.concatenate(parts) * 32767).astype("<i2")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def beep(level: str):
    """Sound on the computer running this code (used by the command-line watcher).
    Windows: real tones through the speakers.  Elsewhere: the terminal bell."""
    try:
        import winsound
        for f, d in BEEPS.get(level, BEEPS["WARNING"]):
            winsound.Beep(f, int(d * 1000))
            time.sleep(0.08)
    except ImportError:
        print("\a" * len(BEEPS.get(level, BEEPS["WARNING"])), end="", flush=True)


class StreamMonitor:
    def __init__(self, model=None, scaler=None, baseline=None, localiser=None,
                 ema=0.2, slope_windows=30, warn_p=0.30, fault_p=0.50, horizon_windows=60,
                 min_slope=0.002, drift_warn=2.0, confirm_warning=3, confirm_fault=2,
                 clear_after=10, max_history=50_000):
        if model is None or scaler is None:
            model, scaler = load_model_and_scaler()
        self.model, self.scaler = model, scaler
        b = baseline or load_baseline()
        self.mu, self.sd, self.drift_ref = b["healthy_mu"], b["healthy_sd"], b["drift_ref"]
        self.localiser = localiser
        self.ema, self.K = ema, slope_windows
        self.warn_p, self.fault_p, self.horizon = warn_p, fault_p, horizon_windows
        self.min_slope, self.drift_warn = min_slope, drift_warn
        self.confirm_warning, self.confirm_fault, self.clear_after = (
            confirm_warning, confirm_fault, clear_after)
        self._max_history = max_history
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self._buf = None
        self._next_start = 0                    # global sample index of the first buffered row
        self.total_samples = 0
        self.n_windows = 0
        self._p_s = None
        self._hist = deque(maxlen=self.K)
        self._recent = deque(maxlen=10)         # scaled features of recent windows
        self._c_warn = self._c_fault = self._below = 0
        self.level = 0
        self.history = deque(maxlen=self._max_history)
        self.alerts = []

    @property
    def level_name(self):
        return NAMES[self.level]

    def history_df(self) -> pd.DataFrame:
        return pd.DataFrame(list(self.history))

    # ------------------------------------------------------------------- feed
    def push(self, rows):
        """Add new samples. Returns (DataFrame of the windows completed by this call,
        list of alerts raised by this call)."""
        rows = np.atleast_2d(np.asarray(rows, float))
        if rows.size == 0:
            return pd.DataFrame(), []
        if rows.shape[1] < 6:
            raise ValueError(f"need 6 signal columns, got {rows.shape[1]}")
        rows = rows[:, :6]
        rows = rows[~np.isnan(rows).any(axis=1)]
        self.total_samples += len(rows)
        self._buf = rows if self._buf is None else np.vstack([self._buf, rows])
        if len(self._buf) < WINDOW_SIZE:
            return pd.DataFrame(), []

        feats = extract_features(self._buf, WINDOW_SIZE, STEP_SIZE)
        n = len(feats)
        Xs = np.nan_to_num(self.scaler.transform(feats))
        p = self.model.predict_proba(Xs)[:, 1]
        drift = drift_score(Xs, self.mu, self.sd) / self.drift_ref
        first_start = self._next_start
        self._buf = self._buf[n * STEP_SIZE:]
        self._next_start += n * STEP_SIZE

        out, new_alerts = [], []
        for i in range(n):
            row, alert = self._step(float(p[i]), float(drift[i]), Xs[i],
                                    first_start + i * STEP_SIZE + WINDOW_SIZE - 1)
            out.append(row)
            if alert:
                new_alerts.append(alert)
        return pd.DataFrame(out), new_alerts

    # ----------------------------------------------------------- one window
    def _step(self, p, drift, x_scaled, sample_index):
        self._p_s = p if self._p_s is None else (1 - self.ema) * self._p_s + self.ema * p
        p_s = self._p_s
        self._hist.append(p_s)
        slope = 0.0
        if len(self._hist) >= 5:
            slope = float(np.polyfit(np.arange(len(self._hist)), np.fromiter(self._hist, float), 1)[0])
        rising = slope > self.min_slope
        eta = max(0.0, (self.fault_p - p_s) / slope) if rising else np.inf
        drift_hit = self.drift_warn is not None and drift > self.drift_warn

        if p_s >= self.fault_p:
            raw = 2
        elif p_s >= self.warn_p or (rising and eta <= self.horizon) or drift_hit:
            raw = 1
        else:
            raw = 0

        # confirmation / hysteresis
        self._c_warn = self._c_warn + 1 if raw >= 1 else 0
        self._c_fault = self._c_fault + 1 if raw >= 2 else 0
        cand = 2 if self._c_fault >= self.confirm_fault else (
            1 if self._c_warn >= self.confirm_warning else 0)
        prev = self.level
        if cand >= self.level:
            self.level, self._below = cand, 0
        else:
            self._below += 1
            if self._below >= self.clear_after:
                self.level, self._below = cand, 0

        self._recent.append(x_scaled)
        row = {"window": self.n_windows, "sample_index": sample_index, "p_fault": p,
               "p_smooth": p_s, "slope": slope, "drift": drift,
               "eta_windows": eta, "eta_samples": eta * STEP_SIZE,
               "raw_state": NAMES[raw], "state": NAMES[self.level]}
        self.history.append(row)
        self.n_windows += 1

        alert = None
        if self.level > prev:
            alert = self._make_alert(row, drift_hit, rising)
            self.alerts.append(alert)
        return row, alert

    def _make_alert(self, row, drift_hit, rising):
        level = NAMES[self.level]
        if level == "FAULT":
            msg = f"FAULT level reached (P(fault) = {row['p_smooth']:.0%})."
        else:
            why = []
            if row["p_smooth"] >= self.warn_p:
                why.append(f"P(fault) = {row['p_smooth']:.0%}")
            if rising and np.isfinite(row["eta_windows"]):
                why.append(f"rising - fault level in ~{row['eta_windows']:.0f} windows "
                           f"(~{row['eta_samples']:.0f} samples)")
            if drift_hit:
                why.append(f"drift from healthy x{row['drift']:.1f}")
            msg = "Fault looks about to occur: " + "; ".join(why or ["abnormal behaviour"]) + "."
        suspect = ""
        if self.localiser is not None and len(self._recent) >= 3:
            try:
                loc = self.localiser.localise(np.array(self._recent))
                suspect = loc["suspect"] if loc["abnormal"] else "no single signal stands out"
            except Exception:
                suspect = ""
        return {"time": datetime.now().strftime("%H:%M:%S"), "level": level,
                "window": row["window"], "sample_index": row["sample_index"],
                "p_smooth": round(row["p_smooth"], 3),
                "eta_windows": None if not np.isfinite(row["eta_windows"]) else round(row["eta_windows"], 1),
                "drift": round(row["drift"], 2), "suspect_signal": suspect, "message": msg}


# ------------------------------------------------------------------ live CSV
class CsvTail:
    """Reads only the NEW rows of a CSV that another program keeps appending to.
    Accepts comma / semicolon / tab / space separated numbers; header lines are skipped;
    extra columns after the 6 signals (e.g. Conditions) are ignored."""
    def __init__(self, path, from_end=True):
        self.path = str(path)
        self.offset = 0
        if from_end:
            try:
                import os
                self.offset = os.path.getsize(self.path)
            except OSError:
                self.offset = 0

    def read_new(self) -> np.ndarray:
        import os
        try:
            size = os.path.getsize(self.path)
        except OSError:
            return np.empty((0, 6))
        if size < self.offset:                      # file was replaced / truncated
            self.offset = 0
        if size == self.offset:
            return np.empty((0, 6))
        with open(self.path, "rb") as f:
            f.seek(self.offset)
            chunk = f.read()
        cut = chunk.rfind(b"\n")
        if cut < 0:                                  # no complete line yet
            return np.empty((0, 6))
        self.offset += cut + 1
        rows = []
        for line in chunk[:cut].decode("utf-8", "ignore").splitlines():
            parts = [t for t in re.split(r"[,;\t ]+", line.strip()) if t]
            try:
                vals = [float(t) for t in parts]
            except ValueError:
                continue                             # header or junk line
            if len(vals) >= 6:
                rows.append(vals[:6])
        return np.array(rows, float).reshape(-1, 6)


# --------------------------------------------------------------------- CLI
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Watch a growing CSV, print warnings and beep.")
    ap.add_argument("--file", required=True, help="CSV your data logger keeps appending to")
    ap.add_argument("--from-start", action="store_true", help="also process rows already in the file")
    ap.add_argument("--poll", type=float, default=0.5, help="seconds between checks")
    ap.add_argument("--no-beep", action="store_true", help="print only, no sound")
    ap.add_argument("--repeat", type=float, default=5.0,
                    help="repeat the beep every N seconds while a warning/fault is active (0 = off)")
    a = ap.parse_args()
    mon = StreamMonitor()
    tail = CsvTail(a.file, from_end=not a.from_start)
    print(f"Watching {a.file} ... (Ctrl+C to stop)")
    last_beep = 0.0
    try:
        while True:
            _, alerts = mon.push(tail.read_new())
            for al in alerts:
                print(f"[{al['time']}] {al['level']:7s} window {al['window']}: {al['message']}"
                      + (f"  Most fault-like signal: {al['suspect_signal']}" if al["suspect_signal"] else ""))
            if not a.no_beep:
                if alerts:
                    beep(max(alerts, key=lambda x: x["level"] == "FAULT")["level"]); last_beep = time.time()
                elif a.repeat and mon.level > 0 and time.time() - last_beep >= a.repeat:
                    beep(mon.level_name); last_beep = time.time()
            time.sleep(a.poll)
    except KeyboardInterrupt:
        print(f"\nStopped. {mon.total_samples:,} samples, {mon.n_windows:,} windows, "
              f"{len(mon.alerts)} alerts.")

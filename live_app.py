"""
LIVE fault monitor  ->  streamlit run live_app.py

Feed data as it arrives and get a warning while a fault is still developing.
Two ways to feed it:
  * "Watch a live CSV file": your data logger / PLC script keeps appending rows to a CSV;
    this app reads only the new rows (checks twice a second).
  * "Replay a recording": plays a file back row by row as if it were live (for demos/testing).
To connect any other source (serial port, socket, MQTT...) call
    StreamMonitor().push(rows)   # rows: (k, 6) array
from your own code - see live_monitor.py.
"""
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import reporting as rp
from early_warning import simulate_degradation
from fault_localization import FaultLocaliser
from live_monitor import CsvTail, StreamMonitor, alert_sound_wav
from monitor_common import (BASELINE_FILE, MODEL_FILE, SCALER_FILE, fault_start_from,
                            load_baseline, load_model_and_scaler, read_signal_file)

st.set_page_config(page_title="Live Fault Monitor", page_icon="📡", layout="wide")

missing = [p.name for p in (MODEL_FILE, SCALER_FILE) if not p.exists()]
if missing:
    st.error("Missing files next to live_app.py: " + ", ".join(missing)); st.stop()
if not BASELINE_FILE.exists():
    st.error("monitor_baseline.joblib not found. Run `python make_baseline.py` once first.")
    st.stop()


@st.cache_resource(show_spinner="Loading your model and healthy baseline...")
def get_resources():
    model, scaler = load_model_and_scaler()
    base = load_baseline()
    return model, scaler, base, FaultLocaliser(base["localiser_stats"], base["feature_names"])


model, scaler, baseline, localiser = get_resources()
S = st.session_state
S.setdefault("mon", None)

# ----------------------------------------------------------------- sidebar
st.sidebar.header("Data source")
source = st.sidebar.radio("Feed the monitor from", ["Watch a live CSV file", "Replay a recording"])
if source == "Watch a live CSV file":
    path = st.sidebar.text_input("Path of the CSV that is being written",
                                 help="Rows need the 6 signal columns in the same order as your "
                                      "training datasets. A header line is fine.")
    from_end = st.sidebar.checkbox("Only look at rows added from now on", True)
    poll = st.sidebar.slider("Check for new rows every (s)", 0.2, 5.0, 0.5, 0.1)
else:
    mode = st.sidebar.radio("Recording", ["One file", "Healthy → faulty blend"],
                            help="The blend slowly mixes your healthy data into your faulty data, "
                                 "to demo an early warning.")
    if mode == "One file":
        up = st.sidebar.file_uploader("Recording (.csv / .xlsx)", type=["csv", "xlsx", "xls"])
        start_row = st.sidebar.number_input("Start at row", 0, 10_000_000, 0, step=1000,
                                            help="The very first rows of your datasets look unlike "
                                                 "the rest; try a later start.")
        healthy_rows = st.sidebar.number_input(
            "Healthy rows at the start (0 = unknown)", 0, 10_000_000, 0,
            help="E.g. 350 if the file is 350 healthy rows then 50 faulty rows. Leave 0 to use "
                 "the file's Condition column, if it has one.")
    else:
        up_h = st.sidebar.file_uploader("Healthy dataset", type=["csv", "xlsx", "xls"], key="h")
        up_f = st.sidebar.file_uploader("Faulty dataset", type=["csv", "xlsx", "xls"], key="f")
        blend_len = st.sidebar.slider("Length (samples)", 5_000, 100_000, 40_000, 5_000)
    chunk = st.sidebar.slider("Samples delivered per step", 20, 2000, 200, 20)
    delay = st.sidebar.slider("Seconds between steps", 0.05, 2.0, 0.3, 0.05)

st.sidebar.header("Warning rules")
warn_p = st.sidebar.slider("Warn when P(fault) ≥", 0.05, 0.90, 0.30, 0.05)
fault_p = st.sidebar.slider("Fault when P(fault) ≥", 0.10, 0.99, 0.50, 0.05)
horizon = st.sidebar.slider("Warn if fault predicted within (windows)", 10, 300, 60, 10)
use_drift = st.sidebar.checkbox("Also warn on drift from healthy", True)
drift_warn = st.sidebar.slider("Drift level (1.0 = edge of healthy)", 1.0, 5.0, 2.0, 0.1,
                               disabled=not use_drift)
confirm = st.sidebar.slider("Windows in a row before a warning", 1, 10, 3,
                            help="Higher = fewer false alarms, slightly later warning.")
st.sidebar.header("Sound")
beep_on = st.sidebar.checkbox("🔊 Beep on warnings", True,
                              help="Two beeps = warning, three higher beeps = fault. Your "
                                   "browser plays it, so keep this tab open and the volume up.")
repeat = st.sidebar.slider("Repeat every (s) while a warning/fault is active (0 = off)",
                           0, 30, 5, disabled=not beep_on)
test_sound = st.sidebar.button("Test sound")
c1, c2 = st.sidebar.columns(2)
start = c1.button("▶ Start", type="primary")
c2.button("■ Stop")

st.title("📡 Live Fault Monitor")
banner_ph, metrics_ph, chart_ph, alerts_ph = st.empty(), st.empty(), st.empty(), st.empty()
sound_ph = st.empty()


def play(level):
    """Play the beep pattern in the browser (autoplay works because you pressed a button)."""
    S["beep_n"] = S.get("beep_n", 0) + 1
    wav = alert_sound_wav(level, S["beep_n"])
    try:
        sound_ph.audio(wav, format="audio/wav", autoplay=True)
    except TypeError:                       # very old Streamlit without autoplay
        sound_ph.audio(wav, format="audio/wav")


if test_sound:
    play("TEST")
    st.sidebar.caption("If you heard nothing: check the volume and that the tab is not muted.")



def render(mon):
    h = mon.history_df()
    if h.empty:
        banner_ph.info("Waiting for data… (needs 20 samples for the first window)")
        return
    last = h.iloc[-1]
    if mon.level == 0:
        banner_ph.success(f"NORMAL - P(fault) {last.p_smooth:.0%}, window {int(last.window)}")
    elif mon.level == 1:
        al = [a for a in mon.alerts if a["level"] == "WARNING"][-1]
        banner_ph.warning(f"⚠ WARNING - {al['message']}"
                          + (f"  Most fault-like signal: {al['suspect_signal']}." if al["suspect_signal"] else ""))
    else:
        banner_ph.error(f"🚨 FAULT - P(fault) {last.p_smooth:.0%}. "
                        f"{len(mon.alerts)} alert(s) so far.")
    with metrics_ph.container():
        a, b, c, d = st.columns(4)
        a.metric("Samples received", f"{mon.total_samples:,}")
        b.metric("Windows", f"{mon.n_windows:,}")
        c.metric("P(fault) now", f"{last.p_smooth:.0%}")
        d.metric("Alerts", len(mon.alerts))
    fig = rp.timeline_figure(h.tail(400).reset_index(drop=True), warn_p, fault_p,
                             S.get("fault_start"))
    chart_ph.pyplot(fig)
    plt.close(fig)
    if mon.alerts:
        alerts_ph.dataframe(pd.DataFrame(mon.alerts).iloc[::-1],
                            hide_index=True)


def new_monitor():
    return StreamMonitor(model, scaler, baseline, localiser, warn_p=warn_p, fault_p=fault_p,
                         horizon_windows=horizon, drift_warn=drift_warn if use_drift else None,
                         confirm_warning=confirm)


if start:
    S["fault_start"] = None
    try:
        if source == "Watch a live CSV file":
            if not path:
                st.warning("Enter the path of the CSV first."); st.stop()
            feed = CsvTail(path, from_end=from_end)
            def next_rows():
                time.sleep(poll)
                return feed.read_new()
        else:
            if mode == "One file":
                if up is None:
                    st.warning("Upload a recording first."); st.stop()
                data, labels = read_signal_file(up, return_labels=True)
                data = data[int(start_row):]
                labels = None if labels is None else labels[int(start_row):]
                S["fault_start"] = fault_start_from(labels, int(healthy_rows))
            else:
                if up_h is None or up_f is None:
                    st.warning("Upload both datasets first."); st.stop()
                h_raw, f_raw = read_signal_file(up_h), read_signal_file(up_f)
                h_raw, f_raw = h_raw[int(len(h_raw) * .8):], f_raw[int(len(f_raw) * .8):]
                data = simulate_degradation(h_raw, f_raw, n=int(blend_len))
                S["fault_start"] = None
            pos = {"i": 0}
            def next_rows():
                time.sleep(delay)
                if pos["i"] >= len(data):
                    return None
                r = data[pos["i"]:pos["i"] + chunk]; pos["i"] += chunk
                return r
        mon = S.mon = new_monitor()
        seen, last_beep = 0, 0.0
        while True:
            rows = next_rows()
            if rows is None:
                break
            mon.push(rows)
            render(mon)
            if len(mon.alerts) > seen:
                for al in mon.alerts[seen:]:
                    st.toast(f"{al['level']}: {al['message']}", icon="🚨" if al["level"] == "FAULT" else "⚠️")
                if beep_on:
                    play("FAULT" if any(x["level"] == "FAULT" for x in mon.alerts[seen:]) else "WARNING")
                    last_beep = time.time()
                seen = len(mon.alerts)
            elif beep_on and repeat and mon.level > 0 and time.time() - last_beep >= repeat:
                play(mon.level_name)
                last_beep = time.time()
        render(mon)
        if S.get("fault_start") is not None and mon.n_windows:
            gt = rp.ground_truth_summary(mon.history_df(), S["fault_start"])
            st.markdown("**Result vs. the real data**\n\n" + "\n".join("- " + ln for ln in gt["lines"]))
        st.caption("Replay finished." if source != "Watch a live CSV file" else "Stopped.")
    except Exception as e:
        st.error(f"Stopped: {e}")
elif S.mon is not None:
    render(S.mon)
    st.caption("Stopped. Press ▶ Start to run again.")
else:
    st.info("Pick a data source in the sidebar and press **▶ Start**.")

# downloads for whatever has been collected
mon = S.mon
if mon is not None and mon.n_windows:
    res = mon.history_df()
    events = rp.build_alert_events(res)
    d1, d2, d3 = st.columns(3)
    d1.download_button("⬇ Alerts log (CSV)", pd.DataFrame(mon.alerts).to_csv(index=False).encode(),
                       "live_alerts.csv", "text/csv", disabled=not mon.alerts)
    d2.download_button("⬇ All windows (CSV)", res.replace(np.inf, np.nan).to_csv(index=False).encode(),
                       "live_windows.csv", "text/csv")
    if d3.button("Build PDF report"):
        S["live_pdf"] = rp.make_pdf_report(res.tail(5000).reset_index(drop=True), events,
                                           localisation=None, source_name="live feed",
                                           model_name=type(model).__name__,
                                           warn_p=warn_p, fault_p=fault_p)
    if S.get("live_pdf"):
        st.download_button("⬇ Download PDF report", S["live_pdf"], "live_report.pdf", "application/pdf")

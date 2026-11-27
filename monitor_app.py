"""
Fault monitoring add-on app  ->  run with:   streamlit run monitor_app.py

Puts the four add-ons in one screen:
  1. live status + EARLY WARNING            (early_warning.py)
  2. which SIGNAL is deviating               (fault_localization.py)
  3. WHY it was flagged (SHAP)               (explain.py)
  4. alert CSV + PDF report downloads        (reporting.py)
Your notebooks and trained models are only LOADED, never changed.
"""
import numpy as np
import pandas as pd
import streamlit as st

import reporting as rp
from early_warning import EarlyWarning, simulate_degradation
from explain import PredictionExplainer
from fault_localization import FaultLocaliser
from monitor_common import (BASELINE_FILE, MODEL_FILE, SCALER_FILE, fault_start_from,
                            load_baseline, load_model_and_scaler, read_signal_file)

st.set_page_config(page_title="Fault Monitoring", page_icon="⚡", layout="wide")


# ----------------------------------------------------------- cached resources
missing = [p.name for p in (MODEL_FILE, SCALER_FILE) if not p.exists()]
if missing:
    st.error("Missing files next to monitor_app.py: " + ", ".join(missing))
    st.stop()
if not BASELINE_FILE.exists():
    st.error("monitor_baseline.joblib not found. Run `python make_baseline.py` once in the "
             "folder that contains X_train_features.csv and y_train.npy (made by your "
             "feature-engineering notebook), then restart this app.")
    st.stop()


@st.cache_resource(show_spinner="Loading your model and healthy baseline...")
def get_resources():
    model, scaler = load_model_and_scaler()
    base = load_baseline()
    return (EarlyWarning(model, scaler, base),
            FaultLocaliser(base["localiser_stats"], base["feature_names"]), base)


@st.cache_resource(show_spinner="Preparing SHAP explainer...")
def get_explainer(_model, _base):
    return PredictionExplainer(_model, _base["shap_background"], _base["feature_names"])


ew, localiser, baseline = get_resources()
model_name = type(ew.model).__name__

# ------------------------------------------------------------------- sidebar
st.sidebar.header("Input")
mode = st.sidebar.radio("Mode", ["Analyse a recording", "Simulate degradation"],
                        help="Simulation blends healthy data into faulty data over time, so "
                             "you can demo the early warning without a run-to-failure recording.")
max_rows = st.sidebar.number_input("Max rows to read", 5_000, 600_000, 100_000, step=5_000,
                                   help="Large Excel files are slow; 100k rows is ~20k windows.")
fs = st.sidebar.number_input("Sampling rate in Hz (0 = unknown)", 0.0, 1e9, 0.0,
                             help="Only used to show times in seconds.") or None

raw = None
source_name = ""
if mode == "Analyse a recording":
    up = st.sidebar.file_uploader("Recording (.csv / .xlsx, 6 signal columns)",
                                  type=["csv", "xlsx", "xls"])
    if up is not None:
        source_name = up.name
    healthy_rows = st.sidebar.number_input(
        "Healthy rows at the start (0 = unknown)", 0, 10_000_000, 0,
        help="If your file is, say, 350 healthy rows followed by 50 faulty rows, enter 350. "
             "Leave 0 to use the file's Condition column (0 healthy / 1 faulty), if it has one. "
             "The dashboard then shows the true healthy/faulty parts and how the system reacted.")
else:
    up_h = st.sidebar.file_uploader("Healthy dataset", type=["csv", "xlsx", "xls"], key="h")
    up_f = st.sidebar.file_uploader("Faulty dataset", type=["csv", "xlsx", "xls"], key="f")
    sim_len = st.sidebar.slider("Simulated length (samples)", 20_000, 200_000, 60_000, 10_000)
    only_test = st.sidebar.checkbox("Use only the last 20 % (unseen test part)", True)
    source_name = "simulated degradation"

st.sidebar.header("Warning rules")
warn_p = st.sidebar.slider("Warning when P(fault) >=", 0.05, 0.90, float(ew.warn_p), 0.05)
fault_p = st.sidebar.slider("Fault when P(fault) >=", 0.10, 0.99, float(ew.fault_p), 0.05)
horizon = st.sidebar.slider("Warn if fault predicted within (windows)", 10, 300,
                            int(ew.horizon), 10)
use_drift = st.sidebar.checkbox("Also warn on drift from healthy", True)
drift_warn = st.sidebar.slider("Drift level (1.0 = edge of healthy)", 1.0, 5.0,
                               float(ew.drift_warn), 0.1, disabled=not use_drift)
run = st.sidebar.button("Analyse", type="primary")

# ------------------------------------------------------------------ run
if run:
    try:
        with st.spinner("Reading data..."):
            if mode == "Analyse a recording":
                if up is None:
                    st.warning("Upload a recording first."); st.stop()
                raw, labels = read_signal_file(up, int(max_rows), return_labels=True)
                ok = ~np.isnan(raw).any(axis=1)
                raw = raw[ok]
                labels = None if labels is None else labels[ok]
                fault_start = fault_start_from(labels, int(healthy_rows))
            else:
                fault_start = None
                if up_h is None or up_f is None:
                    st.warning("Upload both the healthy and the faulty dataset."); st.stop()
                h = read_signal_file(up_h, int(max_rows) * 5 if not only_test else None)
                f = read_signal_file(up_f, int(max_rows) * 5 if not only_test else None)
                if only_test:
                    h, f = h[int(len(h) * .8):], f[int(len(f) * .8):]
                # keep the simulated stream no longer than the source, so the frequency-axis
                # columns stay inside the range the model has seen
                raw = simulate_degradation(h, f, n=min(int(sim_len), len(h)))
            raw = raw[~np.isnan(raw).any(axis=1)] if mode != "Analyse a recording" else raw

        ew.warn_p, ew.fault_p, ew.horizon = warn_p, fault_p, horizon
        ew.drift_warn = drift_warn if use_drift else float('inf')
        with st.spinner(f"Extracting features and predicting ({len(raw):,} samples)..."):
            res = ew.analyse(raw)
            Xs = ew.last_Xs
        flagged = np.where(res.state.to_numpy() != "NORMAL")[0]
        suspects = localiser.per_window_suspect(Xs)
        loc = localiser.localise(Xs, rows=flagged if len(flagged) else None)
        events = rp.build_alert_events(res, suspects, fs)
        st.session_state.update(res=res, Xs=Xs, flagged=flagged, loc=loc, events=events,
                                suspects=suspects, source=source_name, fs=fs, fault_start=fault_start, n_rows=len(raw),
                                warn_p=warn_p, fault_p=fault_p, expl=None, pdf=None)
    except Exception as e:
        st.error(f"Could not analyse the data: {e}")
        st.stop()

# ------------------------------------------------------------------ display
st.title("⚡ Fault Monitoring")
S = st.session_state
if "res" not in S:
    st.info("Choose a mode in the sidebar, upload data and press **Analyse**.")
    st.stop()

res, Xs, flagged, loc, events = S["res"], S["Xs"], S["flagged"], S["loc"], S["events"]
status = rp.overall_status(res)
lt = rp.lead_time(res)

banner = {"NORMAL": st.success, "WARNING": st.warning, "FAULT": st.error}[status]
if status == "NORMAL":
    banner("NORMAL - no warning or fault in this recording.")
elif status == "WARNING":
    w = res.loc[flagged[0]]
    eta = "" if not np.isfinite(w.eta_windows) else f" Estimated ~{w.eta_windows:.0f} windows to fault."
    banner(f"WARNING - a fault looks about to occur (first at window {int(w.window)}, row {int(w.sample_index) + 1}).{eta}")
else:
    fw = res.index[res.state == "FAULT"][0]
    banner(f"FAULT - fault level reached at window {int(res.window[fw])} (row {int(res.sample_index[fw]) + 1})."
           + ("" if lt is None else f" It was warned {lt} windows earlier."))

c1, c2, c3, c4 = st.columns(4)
c1.metric("Windows analysed", f"{len(res):,}")
c2.metric("Alert episodes", len(events))
c3.metric("Peak P(fault)", f"{res.p_smooth.max():.0%}")
c4.metric("Warning → fault lead", "n/a" if lt is None else f"{lt} windows")

tab1, tab2, tab3, tab4 = st.tabs(["1 · Early warning", "2 · Which signal",
                                  "3 · Why (SHAP)", "4 · Alerts & report"])

with tab1:
    if S.get("fault_start") is not None:
        gt = rp.ground_truth_summary(res, S["fault_start"], S.get("n_rows"))
        st.markdown("\n".join("- " + ln for ln in gt["lines"]))
    st.pyplot(rp.timeline_figure(res, S["warn_p"], S["fault_p"], S.get("fault_start")))
    st.caption("Shaded: yellow = warning, red = fault. The warning can fire before the "
               "fault level because it also watches the rising trend and drift from healthy.")
    with st.expander("Per-window table"):
        st.dataframe(res.replace(np.inf, np.nan).round(4))

with tab2:
    st.subheader("Which signal looks most like the fault?")
    if loc["abnormal"]:
        st.write(f"**{loc['suspect']}** is the most fault-like signal "
                 f"({loc['suspect_score']:.0%} of the typical training fault"
                 + (", over the flagged windows)" if len(flagged) else ")")
                 + (". Also abnormal: " + ", ".join(s for s in loc["abnormal_signals"]
                                                     if s != loc["suspect"])
                    if len(loc["abnormal_signals"]) > 1 else ""))
    else:
        st.write("All signals look healthy (below 50 % fault-likeness).")
    st.pyplot(rp.channel_figure(loc["ranking"]))
    st.caption("0 % = behaves like healthy data, 100 % = as abnormal as the faulty training "
               "data. Your data has a single fault class, so this ranks *signals*; it does "
               "not name a fault type."
               + (f" Left out (the training fault leaves no clear trace there): "
                  f"{', '.join(loc['ignored_signals'])}." if loc["ignored_signals"] else ""))
    k = st.slider("Window to inspect", 0, len(res) - 1,
                  int(flagged[0]) if len(flagged) else int(res.p_smooth.idxmax()), key="loc_w")
    st.dataframe(localiser.top_features(Xs[k], 8).round(3))

with tab3:
    st.subheader("Why was a window flagged?")
    default_w = int(flagged[0]) if len(flagged) else int(res.p_smooth.idxmax())
    w_sel = st.slider("Window to explain", 0, len(res) - 1, default_w, key="shap_w")
    if type(ew.model).__name__.startswith("MLP"):
        st.caption("The MLP is not a tree model, so Kernel SHAP is used - a few seconds per window.")
    if st.button("Explain this window"):
        with st.spinner("Computing SHAP values..."):
            explainer = get_explainer(ew.model, baseline)
            expl = explainer.explain_window(Xs[w_sel], top_k=10)
            S["expl"] = (w_sel, expl, explainer.sentence(expl, float(res.p_fault[w_sel])))
            S["pdf"] = None
    if S.get("expl"):
        w_done, expl, text = S["expl"]
        st.markdown(f"**Window {w_done}** - {text}")
        st.pyplot(rp.shap_figure(expl))
        st.dataframe(expl.round(4))

with tab4:
    st.subheader("Alert episodes")
    if len(events):
        st.dataframe(events)
    else:
        st.write("No alerts were raised.")
    d1, d2 = st.columns(2)
    d1.download_button("⬇ Alerts (CSV)", events.to_csv(index=False).encode(),
                       "alerts.csv", "text/csv")
    d2.download_button("⬇ Full per-window results (CSV)",
                       res.replace(np.inf, np.nan).to_csv(index=False).encode(),
                       "window_results.csv", "text/csv")
    st.divider()
    if st.button("Build PDF report"):
        with st.spinner("Building PDF..."):
            ex = S.get("expl")
            S["pdf"] = rp.make_pdf_report(
                res, events, localisation=loc,
                explanation=None if not ex else ex[1],
                explanation_text="" if not ex else ex[2],
                explained_window=None if not ex else ex[0], fault_start=S.get("fault_start"),
                source_name=S["source"], model_name=model_name, fs=S["fs"],
                warn_p=S["warn_p"], fault_p=S["fault_p"])
    if S.get("pdf"):
        st.download_button("⬇ Download PDF report", S["pdf"], "fault_report.pdf",
                           "application/pdf")
        if not S.get("expl"):
            st.caption("Tip: explain a window in tab 3 first and it is included in the PDF.")

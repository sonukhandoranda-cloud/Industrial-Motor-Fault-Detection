"""
Feature 4 - ALERTS & REPORTS (CSV + PDF)

build_alert_events()  -> one row per continuous WARNING / FAULT episode
make_pdf_report()     -> PDF summary (status, alerts, charts, explanation)
Both work on the table returned by EarlyWarning.analyse() in early_warning.py.
"""
import io
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STATE_COLOR = {"NORMAL": "#2e7d32", "WARNING": "#f9a825", "FAULT": "#c62828"}


# ------------------------------------------------------------------ alerts
def build_alert_events(res: pd.DataFrame, suspects=None, fs=None, merge_gap=5) -> pd.DataFrame:
    """Group consecutive non-NORMAL windows into events.
    res      : output of EarlyWarning.analyse()
    suspects : optional array with the suspect signal name for each window
    fs       : sampling rate in Hz (optional) -> adds time in seconds
    merge_gap: events separated by <= this many NORMAL windows are merged
    """
    cols = ["event_id", "level", "start_window", "end_window", "start_sample", "end_sample",
            "duration_windows", "peak_p_fault", "max_drift", "min_eta_windows",
            "suspect_signal"]
    flagged = np.where(res.state.to_numpy() != "NORMAL")[0]
    if len(flagged) == 0:
        return pd.DataFrame(columns=cols)

    groups, start, prev = [], flagged[0], flagged[0]
    for i in flagged[1:]:
        if i - prev > merge_gap + 1:
            groups.append((start, prev)); start = i
        prev = i
    groups.append((start, prev))

    rows = []
    for k, (a, b) in enumerate(groups, 1):
        seg = res.iloc[a:b + 1]
        eta = seg.eta_windows.replace(np.inf, np.nan).min()
        suspect = ""
        if suspects is not None:
            suspect = pd.Series(np.asarray(suspects)[a:b + 1]).mode().iloc[0]
        row = {"event_id": k,
               "level": "FAULT" if (seg.state == "FAULT").any() else "WARNING",
               "start_window": int(seg.window.iloc[0]), "end_window": int(seg.window.iloc[-1]),
               "start_sample": int(seg.sample_index.iloc[0]),
               "end_sample": int(seg.sample_index.iloc[-1]),
               "duration_windows": int(b - a + 1),
               "peak_p_fault": round(float(seg.p_smooth.max()), 4),
               "max_drift": round(float(seg.drift.max()), 3),
               "min_eta_windows": None if np.isnan(eta) else round(float(eta), 1),
               "suspect_signal": suspect}
        if fs:
            row["start_time_s"] = round(row["start_sample"] / fs, 4)
            row["end_time_s"] = round(row["end_sample"] / fs, 4)
        rows.append(row)
    return pd.DataFrame(rows)


def lead_time(res: pd.DataFrame):
    """Windows between the first WARNING and the first FAULT (None if not both)."""
    w = res.index[res.state != "NORMAL"]
    f = res.index[res.state == "FAULT"]
    if len(w) == 0 or len(f) == 0:
        return None
    return int(res.window[f[0]] - res.window[w[0]])


def overall_status(res: pd.DataFrame) -> str:
    if (res.state == "FAULT").any():
        return "FAULT"
    if (res.state == "WARNING").any():
        return "WARNING"
    return "NORMAL"


# ------------------------------------------------------------------ charts
def timeline_figure(res: pd.DataFrame, warn_p=0.30, fault_p=0.50, fault_start=None):
    """fault_start = index (0-based) of the first truly faulty row, if known: draws a green
    'actually healthy' / red 'actually faulty' strip along the top and a line at the change."""
    fig, ax = plt.subplots(figsize=(8, 3.4))
    x = res.sample_index.to_numpy() + 1                      # row number at the end of the window
    ax.plot(x, res.p_fault, color="#90a4ae", lw=0.7, label="fault probability")
    ax.plot(x, res.p_smooth, color="#0d47a1", lw=1.8, label="smoothed")
    for lvl, thr in (("WARNING", warn_p), ("FAULT", fault_p)):
        ax.axhline(thr, color=STATE_COLOR[lvl], ls="--", lw=1, label=f"{lvl.lower()} level")
    for lvl in ("WARNING", "FAULT"):
        m = (res.state == lvl).to_numpy()
        if m.any():
            ax.fill_between(x, 0, 1, where=m, color=STATE_COLOR[lvl], alpha=0.12, step="mid")
    ax.set_ylim(0, 1.22); ax.set_xlabel("row (end of window)"); ax.set_ylabel("P(fault)")
    ax.grid(alpha=0.2)
    if fault_start is not None:
        lo, hi = ax.get_xlim()
        edge = fault_start + 0.5
        last = int(x[-1])
        if edge > lo:
            ax.axvspan(lo, min(edge, hi), ymin=0.86, ymax=1, color="#2e7d32", alpha=0.9, lw=0)
            ax.text((lo + min(edge, hi)) / 2, 1.135, f"ACTUAL: healthy\nrows 1-{fault_start}",
                    ha="center", va="center", color="white", fontsize=6.5, fontweight="bold", linespacing=1.1)
        if edge < hi:
            ax.axvspan(max(edge, lo), hi, ymin=0.86, ymax=1, color="#c62828", alpha=0.9, lw=0)
            ax.text((max(edge, lo) + hi) / 2, 1.135, f"ACTUAL: faulty\nrows {fault_start + 1}-{last}",
                    ha="center", va="center", color="white", fontsize=6.5, fontweight="bold", linespacing=1.1)
        if lo < edge < hi:
            ax.axvline(edge, color="k", ls=":", lw=1.2)
        ax.set_xlim(lo, hi)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.85), fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


def ground_truth_summary(res: pd.DataFrame, fault_start: int, n_rows: int = None) -> dict:
    """Compare the system's reaction with what is really in the file.
    fault_start = index (0-based) of the first faulty row; rows before it are healthy."""
    s = res.sample_index.to_numpy()
    flagged = (res.state != "NORMAL").to_numpy()
    n_rows = n_rows or int(s[-1]) + 1
    healthy_w = s < fault_start                              # windows containing healthy rows only
    n_h, false_w = int(healthy_w.sum()), int((flagged & healthy_w).sum())
    after = np.where(~healthy_w & flagged)[0]
    already = bool(n_h and flagged[healthy_w][-1])
    fault_idx = np.where(~healthy_w & (res.state == "FAULT").to_numpy())[0]

    lines = [f"Actual data: rows 1-{fault_start} healthy, rows {fault_start + 1}-{n_rows} faulty."]
    if n_h:
        lines.append(f"Healthy part: {false_w} of {n_h} windows flagged (false alarms)." if false_w
                     else f"Healthy part: no false alarms ({n_h} windows all NORMAL).")
    out = {"false_alarm_windows": false_w, "healthy_windows": n_h, "missed": len(after) == 0,
           "already_warning": already, "rows_to_warning": None, "rows_to_fault": None}
    if len(after) == 0:
        lines.append("Faulty part: no warning or fault was raised - the fault was MISSED.")
    elif already:
        lines.append("A warning was already active when the first faulty row arrived.")
        out["rows_to_warning"] = 0
    else:
        k = int(s[after[0]]) + 1 - fault_start
        out["rows_to_warning"] = k
        lines.append(f"Warning raised after {k} faulty row{'s' if k != 1 else ''} "
                     f"(row {int(s[after[0]]) + 1}).")
    if len(fault_idx):
        k = int(s[fault_idx[0]]) + 1 - fault_start
        out["rows_to_fault"] = k
        lines.append(f"Fault level reached after {k} faulty rows (row {int(s[fault_idx[0]]) + 1}).")
    elif len(after):
        lines.append("The fault level was not reached.")
    out["lines"] = lines
    return out


def channel_figure(ranking: pd.Series):
    """ranking = fault-likeness per signal (0 = healthy, 1 = as abnormal as training faults)"""
    fig, ax = plt.subplots(figsize=(6, 2.6))
    r = ranking.sort_values()
    ax.barh(r.index, r.values * 100, color=["#c62828" if v >= 0.5 else "#2e7d32" for v in r.values])
    ax.axvline(50, color="k", ls="--", lw=0.8)
    ax.axvline(100, color="k", ls=":", lw=0.8)
    ax.set_xlabel("fault-likeness %   (0 = healthy, 100 = typical training fault)")
    fig.tight_layout()
    return fig


def shap_figure(expl: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    e = expl.iloc[::-1]
    ax.barh(e.feature, e.shap, color=["#c62828" if v > 0 else "#1565c0" for v in e.shap])
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("SHAP value  (red = towards FAULT, blue = towards HEALTHY)")
    ax.tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    return fig


def _png(fig, dpi=160) -> io.BytesIO:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    return buf


# --------------------------------------------------------------------- PDF
def make_pdf_report(res: pd.DataFrame, events: pd.DataFrame, localisation: dict = None,
                    explanation: pd.DataFrame = None, explanation_text: str = "",
                    explained_window: int = None, fault_start: int = None, source_name: str = "uploaded data",
                    model_name: str = "", fs=None, warn_p=0.30, fault_p=0.50) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import (Image, Paragraph, SimpleDocTemplate, Spacer, Table,
                                    TableStyle)

    ss = getSampleStyleSheet()
    out = io.BytesIO()
    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm,
                            topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                            title="Fault monitoring report")
    st = overall_status(res)
    lt = lead_time(res)
    first_w = res.index[res.state != "NORMAL"]
    first_f = res.index[res.state == "FAULT"]

    story = [Paragraph("Fault Monitoring Report", ss["Title"]),
             Paragraph(f"Generated {datetime.now():%d %b %Y, %H:%M} &nbsp;|&nbsp; "
                       f"Source: {source_name}" + (f" &nbsp;|&nbsp; Model: {model_name}" if model_name else ""),
                       ss["Normal"]), Spacer(1, 10)]

    col = STATE_COLOR[st]
    story.append(Paragraph(f'Overall status: <font color="{col}"><b>{st}</b></font>', ss["Heading2"]))

    def fmt(i):
        if len(i) == 0:
            return "none"
        r = res.loc[i[0]]
        t = f" (t = {r.sample_index / fs:.3f} s)" if fs else ""
        return f"window {int(r.window)}, sample {int(r.sample_index)}{t}"

    summary = [["Windows analysed", f"{len(res):,}"],
               ["First warning", fmt(first_w)],
               ["First fault", fmt(first_f)],
               ["Warning-to-fault lead time", "n/a" if lt is None else f"{lt} windows"],
               ["Alert episodes", str(len(events))],
               ["Peak smoothed fault probability", f"{res.p_smooth.max():.1%}"],
               ["Thresholds", f"warning P >= {warn_p:.2f}, fault P >= {fault_p:.2f}"]]
    if localisation:
        summary.append(["Most fault-like signal",
                        f"{localisation['suspect']} ({localisation['suspect_score']:.0%} fault-likeness)"
                        if localisation["abnormal"] else "none - all signals look healthy"])
    t = Table(summary, colWidths=[6 * cm, 11 * cm])
    t.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                           ("BACKGROUND", (0, 0), (0, -1), colors.whitesmoke),
                           ("FONTSIZE", (0, 0), (-1, -1), 9)]))
    story += [t, Spacer(1, 10)]
    if fault_start is not None:
        for ln in ground_truth_summary(res, fault_start)["lines"]:
            story.append(Paragraph("&bull; " + ln, ss["Normal"]))
        story.append(Spacer(1, 6))
    story += [Paragraph("Fault probability over time", ss["Heading3"]),
              Image(_png(timeline_figure(res, warn_p, fault_p, fault_start)),
                    width=17 * cm, height=7.2 * cm)]

    if localisation is not None:
        story += [Paragraph("Which signal looks most like the fault?", ss["Heading3"]),
                  Image(_png(channel_figure(localisation["ranking"])), width=12 * cm, height=5.2 * cm)]

    if explanation is not None and len(explanation):
        title = "Why was this flagged? (SHAP)"
        if explained_window is not None:
            title += f" - window {explained_window}"
        story += [Paragraph(title, ss["Heading3"]),
                  Paragraph(explanation_text, ss["Normal"]),
                  Image(_png(shap_figure(explanation)), width=13 * cm, height=6.8 * cm)]

    story.append(Paragraph("Alert episodes", ss["Heading3"]))
    if len(events):
        keep = ["event_id", "level", "start_window", "end_window", "duration_windows",
                "peak_p_fault", "min_eta_windows", "suspect_signal"]
        data = [keep] + events[keep].head(25).fillna("-").astype(str).values.tolist()
        et = Table(data, repeatRows=1)
        et.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.25, colors.lightgrey),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
                                ("FONTSIZE", (0, 0), (-1, -1), 7)]))
        story.append(et)
        if len(events) > 25:
            story.append(Paragraph(f"... {len(events) - 25} more in the CSV export.", ss["Italic"]))
    else:
        story.append(Paragraph("No warnings or faults were raised.", ss["Normal"]))

    story += [Spacer(1, 10), Paragraph(
        "Note: early warnings are based on the trend of the model's fault probability and on "
        "drift from the healthy baseline. They indicate a developing fault only when the input "
        "is a time-ordered recording; thresholds should be tuned on real degradation data.",
        ss["Italic"])]
    doc.build(story)
    return out.getvalue()

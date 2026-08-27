"""
Industrial Motor Health Monitoring Dashboard
=====================================
Streamlit app that sits on top of the artifacts produced by:
  01_feature_engineering.ipynb  -> fault_scaler.joblib, feature_names.joblib,
                                    X_test_features.csv, y_test.npy
  02_model_training.ipynb       -> lgbm_model.joblib, rf_model.joblib,
                                    mlp_model.joblib, *_training_time.joblib

Run with:
    streamlit run app.py
"""

import os
import time
import base64
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve
)

# ------------------------------------------------------------------
# PAGE CONFIG + BASIC THEME
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Motor Health Monitoring Dashboard",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

def _get_base64_of_file(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()
    return None


_bg_b64 = _get_base64_of_file("background.jpg")

INDUSTRIAL_BG_URL = (
    f"data:image/jpeg;base64,{_bg_b64}" if _bg_b64
    # fallback if background.jpg isn't in the same folder as app.py
    else "https://images.unsplash.com/photo-1518709268805-4e9042af2176?auto=format&fit=crop&w=1600&q=80"
)

st.markdown(f"""
<style>
    /* Industrial background image behind the whole app, darkened for readability */
    .stApp {{
        background-image: linear-gradient(rgba(6,10,18,0.30), rgba(6,10,18,0.30)), url('{INDUSTRIAL_BG_URL}');
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
        background-repeat: no-repeat;
    }}
    section[data-testid="stSidebar"] {{
        background-image: linear-gradient(rgba(5,8,15,0.35), rgba(5,8,15,0.45)), url('{INDUSTRIAL_BG_URL}');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        border-right: 1px solid #22355c;
    }}

    .metric-card {{
        background-color: rgba(18,32,58,0.78);
        backdrop-filter: blur(6px);
        border: 1px solid #2a3f66;
        border-left: 4px solid #3b82f6;
        border-radius: 10px;
        padding: 18px 20px;
        margin-bottom: 10px;
    }}
    .metric-label {{ color: #9db3d6; font-size: 13px; }}
    .metric-value {{ color: #ffffff; font-size: 26px; font-weight: 700; }}
    h1, h2, h3, h4, p, span, label, div {{ color: #e7edf7; }}

    /* Make sidebar content a full-height flex column so the About card can be pushed down */
    section[data-testid="stSidebar"] > div:first-child {{
        display: flex;
        flex-direction: column;
        min-height: 100vh;
    }}

    /* Sidebar "About Project" card */
    .sidebar-about-card {{
        background-color: rgba(18,32,58,0.82);
        backdrop-filter: blur(6px);
        border: 1px solid #2a3f66;
        border-left: 5px solid #22c55e;
        border-radius: 12px;
        padding: 32px 24px;
        margin-top: auto;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        min-height: 420px;
    }}
    .sidebar-about-title {{ color: #22c55e; font-size: 19px; font-weight: 700; margin-bottom: 14px; }}
    .sidebar-about-text {{ color: #b7c4dc; font-size: 14px; line-height: 1.8; margin-bottom: 16px; }}
    .sidebar-about-models {{ color: #60a5fa; font-size: 14px; font-weight: 600; }}

    /* Green-bordered, blue-background section boxes (Upload Dataset / Prediction) */
    div[data-testid="stVerticalBlockBorderWrapper"],
    div.st-key-upload_dataset_box,
    div.st-key-prediction_box {{
        border: 3px solid #22c55e !important;
        border-radius: 14px !important;
        background-color: rgba(18,32,58,0.9) !important;
        backdrop-filter: blur(6px) !important;
        padding: 10px 10px 14px 10px !important;
    }}

    /* Prediction result box */
    .prediction-box {{
        border-radius: 16px;
        padding: 30px 20px;
        text-align: center;
        margin: 10px 6px 6px 6px;
        backdrop-filter: blur(6px);
    }}
    .prediction-shield {{ font-size: 36px; margin-bottom: 6px; }}
    .prediction-status-label {{ font-size: 12px; letter-spacing: 1.5px; text-transform: uppercase; }}
    .prediction-status-value {{ font-size: 42px; font-weight: 800; margin: 4px 0 16px; }}
    .prediction-conf-label {{ font-size: 12px; letter-spacing: 1px; text-transform: uppercase; color: #9db3d6; }}
    .prediction-conf-value {{ font-size: 22px; font-weight: 700; margin: 4px 0 10px; color: #ffffff; }}
    .conf-bar-outer {{
        width: 100%; height: 9px; background: rgba(255,255,255,0.12);
        border-radius: 6px; overflow: hidden;
    }}
    .conf-bar-inner {{ height: 100%; border-radius: 6px; }}
</style>
""", unsafe_allow_html=True)

MODEL_FILES = {
    "LightGBM": "lgbm_model.joblib",
    "Random Forest": "rf_model.joblib",
    "MLP": "mlp_model.joblib",
}
TIME_FILES = {
    "LightGBM": "lgbm_training_time.joblib",
    "Random Forest": "rf_training_time.joblib",
    "MLP": "mlp_training_time.joblib",
}

# ------------------------------------------------------------------
# CACHED LOADERS
# ------------------------------------------------------------------
@st.cache_resource
def load_artifact(path):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_npy(path):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True)
    return None

def load_all_models():
    models = {}
    for name, fname in MODEL_FILES.items():
        m = load_artifact(fname)
        if m is not None:
            models[name] = m
    return models

def load_all_times():
    times = {}
    for name, fname in TIME_FILES.items():
        t = load_artifact(fname)
        if t is not None:
            times[name] = t
    return times

scaler = load_artifact("fault_scaler.joblib")
feature_names = load_artifact("feature_names.joblib")
X_test_df = load_csv("X_test_features.csv")
y_test = load_npy("y_test.npy")
models = load_all_models()
training_times = load_all_times()

CLASS_NAMES = {0: "Healthy", 1: "Faulty"}

# ------------------------------------------------------------------
# FEATURE EXTRACTION (mirrors 01_feature_engineering.ipynb)
# ------------------------------------------------------------------
def extract_statistical_features(dataframe, window_size=20, step_size=5, label_col="Condition"):
    has_label = label_col in dataframe.columns
    X = dataframe.drop(columns=[label_col]).values if has_label else dataframe.values
    y = dataframe[label_col].values if has_label else None

    X_engineered, labels = [], []
    n_signals = X.shape[1]

    for start in range(0, len(dataframe) - window_size + 1, step_size):
        window = X[start:start + window_size]
        feature_vector = []
        for col in range(n_signals):
            signal = window[:, col]
            mean_, std_ = np.mean(signal), np.std(signal)
            rms = np.sqrt(np.mean(signal ** 2))
            sig_min, sig_max = np.min(signal), np.max(signal)
            pkpk = sig_max - sig_min
            sig_skew, sig_kurt = skew(signal), kurtosis(signal)
            fft_vals = np.abs(np.fft.rfft(signal))
            fft_freqs = np.fft.rfftfreq(len(signal))
            dom_idx = np.argmax(fft_vals[1:]) + 1 if len(fft_vals) > 1 else 0
            dom_freq, dom_mag = fft_freqs[dom_idx], fft_vals[dom_idx]
            feature_vector.extend([mean_, std_, rms, sig_min, sig_max, pkpk,
                                    sig_skew, sig_kurt, dom_freq, dom_mag])
        X_engineered.append(feature_vector)
        if has_label:
            labels.append(y[start + window_size - 1])

    return np.array(X_engineered), (np.array(labels) if has_label else None)

# ------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Industrial Motor\nHealth Monitoring")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Model Comparison", "Evaluation", "Data Preview", "About Project"],
    label_visibility="collapsed",
)

missing_core = [n for n in ["fault_scaler.joblib", "feature_names.joblib"] if not os.path.exists(n)]
if missing_core:
    st.sidebar.warning(f"Missing: {', '.join(missing_core)}. Some features will not work until these files are placed in the app.py folder.")

_sidebar_feature_count = len(feature_names) if feature_names is not None else "N"
st.sidebar.markdown("<div style='height:70px'></div>", unsafe_allow_html=True)
st.sidebar.markdown(
    f"""<div class="sidebar-about-card">
    <div class="sidebar-about-title">About Project</div>
    <div class="sidebar-about-text">This project uses machine learning models to detect
    faults in industrial systems using {_sidebar_feature_count} engineered features.</div>
    <div class="sidebar-about-models">Models used: Random Forest, LightGBM, MLP</div>
    </div>""",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# HOME PAGE
# ------------------------------------------------------------------
if page == "Home":
    st.title("Industrial Motor Health Monitoring Dashboard")
    st.caption("ML-powered system to detect faults in industrial equipment using engineered features")

    # session_state holders so other pages (Prediction) can reuse the same upload
    if "raw_df" not in st.session_state:
        st.session_state.raw_df = None
    if "X_feat" not in st.session_state:
        st.session_state.X_feat = None

    total_features = len(feature_names) if feature_names is not None else 0

    # Best model + its accuracy on the held-out test set
    best_model_name, best_accuracy = None, None
    if models and X_test_df is not None and y_test is not None:
        scores = {n: f1_score(y_test, m.predict(X_test_df), average="weighted", zero_division=0) for n, m in models.items()}
        best_model_name = max(scores, key=scores.get)
        best_accuracy = accuracy_score(y_test, models[best_model_name].predict(X_test_df))

    # Total dataset records (fixed display value)
    total_records = 1048578

    st.markdown("""
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.2/css/all.min.css">
""", unsafe_allow_html=True)
    
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#22c55e;"><div class="metric-label">
    <i class="fa-solid fa-database" style="color:#22c55e !important;font-size:18px;"></i>
    <span>Total Dataset Records</span>
</div>
        <div class="metric-value">{total_records:,}</div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#a855f7;"><div class="metric-label">
    <i class="fa-solid fa-gear" style="color:#22c55e !important;font-size:18px;"></i>
    <span>Feature Engineered</span>
</div>
        <div class="metric-value">{total_features}</div></div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class="metric-card" style="border-left-color:#22c55e;"><div class="metric-label">
    <i class="fa-solid fa-trophy" style="color:#22c55e !important;font-size:18px;"></i>
    <span>Best Model</span>
</div>
        <div class="metric-value" style="font-size:20px;">{best_model_name or "-"}</div></div>""", unsafe_allow_html=True)
    with c4:
        acc_display = f"{best_accuracy:.2%}" if best_accuracy is not None else "-"
        st.markdown(f"""<div class="metric-card" style="border-left-color:#10b981;"><div class="metric-label">Accuracy</div>
        <div class="metric-value">{acc_display}</div></div>""", unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns(2)

    # ---------------- Upload Dataset card ----------------
    with col_left:
        with st.container(border=True, key="upload_dataset_box"):
            st.subheader("📂 Upload Dataset")
            home_upload = st.file_uploader(
                "Drag & drop or browse (.csv / .xlsx)", type=["csv", "xlsx"], key="home_uploader"
            )
            if home_upload is not None:
                raw_df = pd.read_csv(home_upload) if home_upload.name.endswith(".csv") else pd.read_excel(home_upload)

                if "Condition" not in raw_df.columns:
                    cols = [f"Feature_{i+1}" for i in range(raw_df.shape[1] - 1)] + ["Condition"]
                    if len(cols) == raw_df.shape[1]:
                        raw_df.columns = cols

                st.session_state.raw_df = raw_df
                st.success(f"Loaded `{home_upload.name}` — shape {raw_df.shape}")
                st.dataframe(raw_df.head(5), use_container_width=True)

                if scaler is not None and st.button("Extract features from this dataset"):
                    X_feat, y_feat = extract_statistical_features(raw_df, 20, 5)
                    st.session_state.X_feat = pd.DataFrame(X_feat, columns=feature_names) if feature_names else pd.DataFrame(X_feat)
                    st.session_state.y_feat = y_feat
                    st.rerun()
            elif st.session_state.raw_df is not None:
                st.caption("Currently loaded:")
                st.dataframe(st.session_state.raw_df.head(5), use_container_width=True)
                if st.button("Clear loaded dataset"):
                    st.session_state.raw_df = None
                    st.session_state.X_feat = None
                    st.rerun()
            else:
                st.caption("No dataset loaded — upload a dataset to run a prediction.")

    # ---------------- Prediction box ----------------
    with col_right:
        with st.container(border=True, key="prediction_box"):
            st.subheader("🎯 Prediction")
            if not models or scaler is None:
                st.info("Model and scaler are required to run a prediction.")
            elif st.session_state.X_feat is None:
                st.caption("Upload a dataset and click 'Extract features from this dataset' — the result will appear here.")
            else:
                active_model = models.get(best_model_name) or list(models.values())[0]
                X_scaled = scaler.transform(st.session_state.X_feat)
                preds = active_model.predict(X_scaled)

                if hasattr(active_model, "predict_proba"):
                    confidence = active_model.predict_proba(X_scaled).max(axis=1)
                else:
                    confidence = np.ones(len(preds))

                majority_pred = int(pd.Series(preds).mode()[0])
                avg_conf = float(np.mean(confidence)) * 100
                status = CLASS_NAMES.get(majority_pred, str(majority_pred))
                color = "#22c55e" if status == "Healthy" else "#ef4444"
                icon = "🛡️" if status == "Healthy" else "⚠️"

                st.markdown(
                    f"""<div class="prediction-box" style="background-color:{color}18;border:2px solid {color};">
                    <div class="prediction-shield">{icon}</div>
                    <div class="prediction-status-label" style="color:{color};">Status</div>
                    <div class="prediction-status-value" style="color:{color};">{status.upper()}</div>
                    <div class="prediction-conf-label">Confidence Score</div>
                    <div class="prediction-conf-value">{avg_conf:.2f}%</div>
                    <div class="conf-bar-outer"><div class="conf-bar-inner" style="width:{avg_conf}%;background:{color};"></div></div>
                    </div>""",
                    unsafe_allow_html=True,
                )

# ------------------------------------------------------------------
# PREDICTION PAGE
# ------------------------------------------------------------------
elif page == "Prediction":
    st.title("🔍 Prediction")

    if not models or scaler is None or feature_names is None:
        st.error("Model, scaler, and feature_names files are required for prediction. Place them in this folder first.")
    else:
        default_model = st.session_state.get("active_model", list(models.keys())[0])
        default_idx = list(models.keys()).index(default_model) if default_model in models else 0
        model_choice = st.selectbox("Select Model", list(models.keys()), index=default_idx)

        home_raw = st.session_state.get("raw_df")
        if home_raw is not None:
            st.info("A dataset is already loaded from the Home page and is being used. Upload a new file to replace it.")
        uploaded = st.file_uploader("Upload raw signal file (.xlsx or .csv) — optional, the Home page dataset will be used if left empty", type=["xlsx", "csv"])

        window_size = st.number_input("Window size", value=20, min_value=2)
        step_size = st.number_input("Step size", value=5, min_value=1)

        raw_df = None
        if uploaded is not None:
            raw_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
            if "Condition" not in raw_df.columns:
                cols = [f"Feature_{i+1}" for i in range(raw_df.shape[1] - 1)] + ["Condition"]
                if len(cols) == raw_df.shape[1]:
                    raw_df.columns = cols
        elif home_raw is not None:
            raw_df = home_raw

        if raw_df is not None:
            st.write("Preview:", raw_df.head())

            if st.button("Run Prediction"):
                X_feat, _ = extract_statistical_features(raw_df, window_size, step_size)
                if X_feat.shape[0] == 0:
                    st.error("Window size is larger than the uploaded data — try a smaller window size.")
                else:
                    X_scaled = scaler.transform(X_feat)
                    model = models[model_choice]
                    preds = model.predict(X_scaled)

                    if hasattr(model, "predict_proba"):
                        probs = model.predict_proba(X_scaled)
                        confidence = probs.max(axis=1)
                    else:
                        confidence = np.ones(len(preds))

                    majority_pred = int(pd.Series(preds).mode()[0])
                    avg_conf = float(np.mean(confidence)) * 100

                    status = CLASS_NAMES.get(majority_pred, str(majority_pred))
                    color = "#22c55e" if status == "Healthy" else "#ef4444"
                    icon = "🛡️" if status == "Healthy" else "⚠️"

                    st.markdown(f"""
                    <div class="prediction-box" style="background-color:{color}18;border:2px solid {color};">
                        <div class="prediction-shield">{icon}</div>
                        <div class="prediction-status-label" style="color:{color};">Status</div>
                        <div class="prediction-status-value" style="color:{color};">{status.upper()}</div>
                        <div class="prediction-conf-label">Confidence Score</div>
                        <div class="prediction-conf-value">{avg_conf:.2f}%</div>
                        <div class="conf-bar-outer"><div class="conf-bar-inner" style="width:{avg_conf}%;background:{color};"></div></div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.write(f"Windows evaluated: {len(preds)}")
                    st.bar_chart(pd.Series(preds).map(CLASS_NAMES).value_counts())
        else:
            st.info("Upload a raw signal file to run a prediction (same format as dataset_Healthy / dataset_Faulty).")

# ------------------------------------------------------------------
# MODEL COMPARISON PAGE
# ------------------------------------------------------------------
elif page == "Model Comparison":
    st.title("📊 Model Performance Overview")

    if not models or X_test_df is None or y_test is None:
        st.error("Models, X_test_features.csv, and y_test.npy are required for comparison.")
    else:
        rows = []
        for name, model in models.items():
            preds = model.predict(X_test_df)
            rows.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, preds),
                "Precision": precision_score(y_test, preds, average="weighted", zero_division=0),
                "Recall": recall_score(y_test, preds, average="weighted", zero_division=0),
                "F1 Score": f1_score(y_test, preds, average="weighted", zero_division=0),
                "Training Time (s)": training_times.get(name, np.nan),
            })
        results_df = pd.DataFrame(rows).sort_values("F1 Score", ascending=False)
        best_model = results_df.iloc[0]["Model"]
        st.success(f"🏆 Best model by F1 Score: **{best_model}**")

        st.dataframe(
            results_df.style.format({
                "Accuracy": "{:.2%}", "Precision": "{:.2%}",
                "Recall": "{:.2%}", "F1 Score": "{:.2%}",
                "Training Time (s)": "{:.2f}"
            }),
            use_container_width=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("F1 Score by Model")
            st.bar_chart(results_df.set_index("Model")["F1 Score"])
        with col2:
            st.subheader("Training Time Comparison")
            st.bar_chart(results_df.set_index("Model")["Training Time (s)"])

# ------------------------------------------------------------------
# EVALUATION PAGE
# ------------------------------------------------------------------
elif page == "Evaluation":
    st.title("📈 Evaluation")

    if not models or X_test_df is None or y_test is None:
        st.error("Models, X_test_features.csv, and y_test.npy are required for evaluation.")
    else:
        model_choice = st.selectbox("Select Model", list(models.keys()))
        model = models[model_choice]
        preds = model.predict(X_test_df)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(f"Confusion Matrix — {model_choice}")
            cm = confusion_matrix(y_test, preds)
            fig, ax = plt.subplots(figsize=(4.5, 4))
            im = ax.imshow(cm, cmap="Blues")
            labels = [CLASS_NAMES.get(i, str(i)) for i in sorted(set(y_test.tolist()))]
            ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
            ax.set_yticks(range(len(labels))); ax.set_yticklabels(labels)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            fig.patch.set_alpha(0)
            st.pyplot(fig)

        with col2:
            st.subheader("Classification Report")
            report = classification_report(y_test, preds, output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose().round(2), use_container_width=True)

        col3, col4 = st.columns(2)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_test_df)[:, 1]

            with col3:
                st.subheader("ROC Curve")
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_auc = auc(fpr, tpr)
                fig, ax = plt.subplots(figsize=(4.5, 4))
                ax.plot(fpr, tpr, color="#22c55e", label=f"AUC = {roc_auc:.2f}")
                ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
                ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
                ax.legend()
                fig.patch.set_alpha(0)
                st.pyplot(fig)

            with col4:
                st.subheader("Precision-Recall Curve")
                prec, rec, _ = precision_recall_curve(y_test, probs)
                pr_auc = auc(rec, prec)
                fig, ax = plt.subplots(figsize=(4.5, 4))
                ax.plot(rec, prec, color="#a855f7", label=f"AP = {pr_auc:.2f}")
                ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
                ax.legend()
                fig.patch.set_alpha(0)
                st.pyplot(fig)
        else:
            st.info(f"{model_choice} does not support predict_proba, so the ROC/PR curves are skipped.")

# ------------------------------------------------------------------
# DATA PREVIEW PAGE
# ------------------------------------------------------------------
elif page == "Data Preview":
    st.title("🗂️ Data Preview")
    if X_test_df is not None:
        st.write("X_train_features.csv shape:", X_test_df.shape)
        st.dataframe(X_test_df.head(50), use_container_width=True)
    else:
        st.info("X_test_features.csv was not found in this folder.")

    uploaded = st.file_uploader("Or preview your own file", type=["csv", "xlsx"], key="preview_uploader")
    if uploaded is not None:
        df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        st.write(df.shape)
        st.dataframe(df.head(50), use_container_width=True)

# ------------------------------------------------------------------
# ABOUT PAGE
# ------------------------------------------------------------------
elif page == "About Project":
    st.title("ℹ️ About Project")
    st.markdown("""
This dashboard accompanies a 3-notebook fault-detection pipeline:

1. **01_feature_engineering.ipynb** — leak-proof train/test split on raw signals,
   sliding-window statistical + FFT feature extraction, scaling.
2. **02_model_training.ipynb** — LightGBM, Random Forest, and MLP, each tuned with
   `RandomizedSearchCV`.
3. **03_inference_evaluation.ipynb** — full metrics, confusion matrices, ROC/PR
   curves, SHAP, and model comparison.

This Streamlit app (`app.py`) loads the saved artifacts from those notebooks and
exposes prediction, comparison, and evaluation as an interactive dashboard.
    """)

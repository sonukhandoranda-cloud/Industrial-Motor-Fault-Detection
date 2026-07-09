# Motor Fault Detection Pipeline (split version)

The original merged notebook is now split into 3 stages + a dashboard, each handing off
saved files to the next.

## Files
| File | Stage | Reads | Writes |
|---|---|---|---|
| `01_feature_engineering.ipynb` | Data processing + feature engineering | `dataset_Healthy (1).xlsx`, `dataset_Faulty (1).xlsx` | `X_train_features.csv`, `X_test_features.csv`, `y_train.npy`, `y_test.npy`, `fault_scaler.joblib`, `feature_names.joblib` |
| `02_model_training.ipynb` | Model training (LightGBM, RF, MLP) | `X_train_features.csv`, `y_train.npy`, `feature_names.joblib` | `lgbm_model.joblib`, `rf_model.joblib`, `mlp_model.joblib`, `*_training_time.joblib` |
| `03_inference_evaluation.ipynb` | Evaluation + comparison | `X_test_features.csv`, `y_test.npy`, `X_train_features.csv`, all 3 `*_model.joblib`, all 3 `*_training_time.joblib` | `fault_model_best.joblib` |
| `app.py` | Streamlit inference dashboard | `fault_model_best.joblib`, `fault_scaler.joblib` | — |

## Setup
```bash
pip install pandas numpy scipy scikit-learn lightgbm shap streamlit joblib openpyxl matplotlib seaborn
```

## Run order
All three notebooks (plus `app.py`) must run **in the same working folder**, since each
one saves files the next one loads — there's no need to re-run earlier steps if only a
later stage changes.

1. **`01_feature_engineering.ipynb`**
   - Loads `dataset_Healthy (1).xlsx` + `dataset_Faulty (1).xlsx` (rename the filenames in
     the "LOAD DATA" cell if yours differ)
   - Leak-proof split: splits the *raw* signal into train/test (80/20, unshuffled) per
     condition, **before** windowing — no test window can share raw rows with a training
     window
   - Slides a window (size=20, step=5) and extracts 10 features per channel: mean, std,
     RMS, min, max, peak-to-peak, skew, kurtosis, dominant FFT frequency, dominant FFT
     magnitude
   - Scales with `StandardScaler`
   - Saves everything needed for training

2. **`02_model_training.ipynb`**
   - Loads the saved training features
   - Trains **LightGBM**, **Random Forest**, and **MLP**, each tuned via
     `RandomizedSearchCV`
   - Saves all three trained models + their training times — no evaluation happens here

3. **`03_inference_evaluation.ipynb`**
   - Loads the saved test features + all three trained models
   - Evaluates **LightGBM fully** (metrics → classification report → confusion matrix →
     ROC curve → PR curve → SHAP summary), then **Random Forest fully**, then **MLP fully**
     (no SHAP — `TreeExplainer` doesn't apply to neural nets)
   - Final section: combined ROC curve across all three + comparison table
   - Saves whichever model scored highest on F1 as `fault_model_best.joblib`

4. **`app.py`** — the dashboard
   ```bash
   streamlit run app.py
   ```
   Copy `fault_model_best.joblib` (from step 3) and `fault_scaler.joblib` (from step 1)
   into the same folder as `app.py` first. Upload a CSV window (same `Feature_1..Feature_N`
   columns, ≥20 consecutive rows) and get prediction + confidence % + SHAP explanation
   (auto-skipped if MLP won).

## Running the Streamlit Dashboard

1. Clone the repository.
2. Install the required packages:
   pip install -r requirements.txt
3. Make sure the required model files (.joblib) are present in the project folder.
4. Run:
   streamlit run app.py

   ## 📸 Dashboard

![Dashboard](Screenshot%202026-07-09%20184313.png)

![Dashboard](Screenshot%202026-07-09%20184330.png)

![Dashboard](Screenshot%202026-07-09%20184349.png)

![Dashboard](Screenshot%202026-07-09%20184723.png)
   
## Notes

- Large datasets (.xlsx), generated feature files (.csv), and NumPy arrays (.npy) are excluded from this repository to keep it lightweight.
- Run the notebooks in the given order to regenerate these files.
- The trained model files (.joblib) are included because they are required to run the Streamlit dashboard.
- 
## Why split it this way
- **Re-run only what changed.** Tweaking model hyperparameters? Just re-run notebook 02 —
  no need to redo feature extraction on 500K+ rows again.
- **Cleaner for a report/viva.** Each notebook has one job and one section of your
  methodology write-up (data prep → training → evaluation) maps directly to one file.
- **Matches how a real ML pipeline is usually structured** — separating data processing,
  training, and evaluation is standard practice, not just for this project's size.

## Things to double check
- **Window size (20) and step size (5)** — tune these based on your actual signal
  sampling rate if you know it.
- **Class balance** — check `pd.Series(y_train).value_counts()` in notebook 01's output.
  If Healthy/Faulty window counts are very unequal, add `class_weight="balanced"` (RF/MLP)
  or `is_unbalance=True` (LightGBM) in notebook 02 before tuning.
- **File paths** — all three notebooks assume they're run in the same folder. If you're
  using Colab, upload the previous stage's output files at the start of each notebook, or
  mount Drive and keep everything in one folder.

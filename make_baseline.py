"""
Run ONCE, in the folder that contains X_train_features.csv and y_train.npy
(the outputs of your feature-engineering notebook):

    python make_baseline.py

It saves a small file, monitor_baseline.joblib (a few hundred KB), holding what the add-ons
need to know about YOUR healthy / faulty training data.  Commit that file to GitHub: the
big CSV/NPY files can stay git-ignored.
"""
import joblib
import numpy as np
import pandas as pd

from fault_localization import FaultLocaliser
from monitor_common import (BASELINE_FILE, SIGNAL_NAMES, MEASURED, XTRAIN_FILE, YTRAIN_FILE,
                            drift_score)

for f in (XTRAIN_FILE, YTRAIN_FILE):
    if not f.exists():
        raise SystemExit(f"{f.name} not found next to make_baseline.py. Run your "
                         "feature-engineering notebook first and keep its outputs here.")

X = pd.read_csv(XTRAIN_FILE)
y = np.load(YTRAIN_FILE)
names = list(X.columns)
assert X.shape[1] == 60 and len(X) == len(y), f"unexpected shapes {X.shape} vs {y.shape}"
assert set(np.unique(y)) == {0, 1}, "y_train must contain both classes 0 and 1"
A = X.to_numpy(float)

healthy = A[y == 0]
mu, sd = healthy.mean(0), healthy.std(0) + 1e-9
drift_ref = float(np.percentile(drift_score(healthy, mu, sd), 99))
stats = FaultLocaliser.fit_stats(A, y, names)
background = X.sample(min(300, len(X)), random_state=42).to_numpy(float)

joblib.dump({"healthy_mu": mu, "healthy_sd": sd, "drift_ref": drift_ref,
             "localiser_stats": stats, "shap_background": background,
             "feature_names": names, "n_train_windows": len(X)}, BASELINE_FILE)

print(f"Saved {BASELINE_FILE.name}")
print(f"  training windows : {len(X):,}  (healthy {int((y == 0).sum()):,}, faulty {int((y == 1).sum()):,})")
print(f"  drift reference  : {drift_ref:.3f}")
print("  how clearly the fault shows in each signal (separation; needs >= 0.5 to be used):")
for i in MEASURED:
    flag = "used" if stats["informative"][i] else "LEFT OUT"
    print(f"    {SIGNAL_NAMES[i]:24s} {stats['separation'][i]:6.2f}   {flag}")

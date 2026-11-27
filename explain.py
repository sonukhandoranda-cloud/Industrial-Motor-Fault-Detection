"""
Feature 3 - WHY was a window flagged?  (SHAP)

Your best model is the MLP, so this uses Kernel SHAP (a few seconds per window).
If you ever switch to a tree model (LightGBM / Random Forest) it automatically
uses the fast TreeExplainer instead.
"""
import numpy as np
import pandas as pd
import shap

from monitor_common import feature_label

TREE_TYPES = ("LGBM", "RandomForest", "XGB", "GradientBoosting", "ExtraTrees", "DecisionTree")


class PredictionExplainer:
    def __init__(self, model, background, names, n_background=25, nsamples=300):
        self.model, self.names, self.nsamples = model, list(names), nsamples
        bg = np.asarray(background, float)
        if any(t in type(model).__name__ for t in TREE_TYPES):
            self.kind = "tree"
            self._explainer = shap.TreeExplainer(model)
        else:
            self.kind = "kernel"
            self._explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1], shap.kmeans(bg, n_background))

    @staticmethod
    def _fault_class(v):
        if isinstance(v, list):
            v = v[1] if len(v) > 1 else v[0]
        v = np.asarray(v)
        if v.ndim == 3:
            v = v[:, :, 1] if v.shape[2] > 1 else v[:, :, 0]
        return v

    def shap_values(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if self.kind == "tree":
            return self._fault_class(self._explainer.shap_values(X))
        return self._fault_class(self._explainer.shap_values(X, nsamples=self.nsamples, silent=True))

    def explain_window(self, x_row, top_k=10) -> pd.DataFrame:
        """positive SHAP = pushes towards FAULT, negative = towards HEALTHY"""
        x_row = np.asarray(x_row, float).reshape(1, -1)
        sv = self.shap_values(x_row)[0]
        idx = np.argsort(np.abs(sv))[::-1][:top_k]
        return pd.DataFrame({"feature": [feature_label(self.names[i]) for i in idx],
                             "value_scaled": x_row[0, idx], "shap": sv[idx],
                             "pushes_towards": np.where(sv[idx] > 0, "FAULT", "HEALTHY")})

    def sentence(self, expl: pd.DataFrame, p_fault: float) -> str:
        up = expl[expl.shap > 0].head(3).feature.tolist()
        if not up:
            return f"Fault probability {p_fault:.0%}: no feature is pushing towards a fault."
        text = f"Fault probability {p_fault:.0%}. Main reasons: " + "; ".join(up) + "."
        if any("freq_axis" in f for f in up):
            text += (" Note: a *_freq_axis feature is the window's position on the frequency "
                     "axis, not a measurement - the model is partly using where in the "
                     "spectrum the window sits.")
        return text

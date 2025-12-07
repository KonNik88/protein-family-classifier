"""
Interpretability helpers (SHAP/LIME) for models on top of ESM embeddings.
"""

from __future__ import annotations

from typing import Any, Literal, Dict

import numpy as np
import shap


def make_shap_explainer(
    model: Any,
    X_background: np.ndarray,
    model_type: Literal["tree", "linear", "kernel"] = "kernel",
):
    """
    Build SHAP explainer for a given model.

    Parameters
    ----------
    model : fitted sklearn-like model
    X_background : np.ndarray
        Background sample for KernelExplainer or similar.
    model_type : {"tree", "linear", "kernel"}

    Returns
    -------
    explainer : shap.Explainer subclass
    """
    if model_type == "tree":
        return shap.TreeExplainer(model)
    if model_type == "linear":
        return shap.LinearExplainer(model, X_background)
    if model_type == "kernel":
        return shap.KernelExplainer(model.predict_proba, X_background)

    raise ValueError(f"Unknown model_type: {model_type}")


def shap_values_for_samples(
    explainer,
    X: np.ndarray,
    max_samples: int = 50,
):
    """
    Compute SHAP values for a subset of samples.
    """
    if X.shape[0] > max_samples:
        X = X[:max_samples]

    shap_vals = explainer.shap_values(X)
    return shap_vals

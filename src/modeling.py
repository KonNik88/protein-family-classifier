"""
Model training & evaluation utilities on top of ESM embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple, Literal, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


ModelType = Literal["logreg", "rf", "mlp"]


@dataclass
class DatasetSplits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    label_encoder: LabelEncoder


def load_embeddings_npz(path: str | Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load embeddings from npz saved by cache_embeddings.

    Returns
    -------
    ids : np.ndarray, shape (N,)
    X : np.ndarray, shape (N, D)
    y : np.ndarray, shape (N,) or None (if not present)
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    ids = data["ids"]
    X = data["X"]
    y = data["y"] if "y" in data.files else None
    return ids, X, y


def build_splits_from_df_and_embeddings(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    emb_train: np.ndarray,
    emb_val: np.ndarray,
    emb_test: np.ndarray,
    label_col: str = "label",
) -> DatasetSplits:
    """
    Align DataFrame labels with corresponding embeddings.
    Assumes order has already been synced.
    """
    y_train_raw = df_train[label_col].astype(str).values
    y_val_raw = df_val[label_col].astype(str).values
    y_test_raw = df_test[label_col].astype(str).values

    le = LabelEncoder()
    le.fit(y_train_raw)

    y_train = le.transform(y_train_raw)
    y_val = le.transform(y_val_raw)
    y_test = le.transform(y_test_raw)

    return DatasetSplits(
        X_train=emb_train,
        y_train=y_train,
        X_val=emb_val,
        y_val=y_val,
        X_test=emb_test,
        y_test=y_test,
        label_encoder=le,
    )


def make_model(model_type: ModelType, random_state: int = 42):
    """
    Factory for simple baseline models.
    """
    if model_type == "logreg":
        return LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight="balanced",
        )
    if model_type == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced_subsample",
        )
    if model_type == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            batch_size=128,
            max_iter=100,
            random_state=random_state,
        )

    raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_model(
    model,
    splits: DatasetSplits,
    split: Literal["val", "test"] = "val",
) -> Dict[str, float]:
    """
    Evaluate model on a given split.
    """
    if split == "val":
        X, y = splits.X_val, splits.y_val
    else:
        X, y = splits.X_test, splits.y_test

    y_pred = model.predict(X)

    metrics = {
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "f1_macro": f1_score(y, y_pred, average="macro"),
    }
    return metrics


def full_report(
    model,
    splits: DatasetSplits,
    split: Literal["val", "test"] = "test",
) -> str:
    """
    Sklearn-style classification report with decoded labels.
    """
    if split == "val":
        X, y = splits.X_val, splits.y_val
    else:
        X, y = splits.X_test, splits.y_test

    y_pred = model.predict(X)
    y_true_labels = splits.label_encoder.inverse_transform(y)
    y_pred_labels = splits.label_encoder.inverse_transform(y_pred)

    report = classification_report(
        y_true_labels,
        y_pred_labels,
        digits=3,
    )
    return report


def confusion_matrix_decoded(
    model,
    splits: DatasetSplits,
    split: Literal["val", "test"] = "test",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Confusion matrix + class labels (decoded).
    """
    if split == "val":
        X, y = splits.X_val, splits.y_val
    else:
        X, y = splits.X_test, splits.y_test

    y_pred = model.predict(X)
    labels = np.arange(len(splits.label_encoder.classes_))
    cm = confusion_matrix(y, y_pred, labels=labels)

    class_names = splits.label_encoder.inverse_transform(labels)
    return cm, class_names


def save_model(model, splits: DatasetSplits, out_path: str | Path) -> Path:
    """
    Save model + label encoder as a joblib dict.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": model, "label_encoder": splits.label_encoder},
        out_path,
    )
    return out_path


def load_model(path: str | Path):
    """
    Load model + label encoder.
    """
    path = Path(path)
    bundle = joblib.load(path)
    return bundle["model"], bundle["label_encoder"]

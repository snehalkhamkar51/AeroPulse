"""Prediction, health classification and explainability helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd

from .config import HEALTH_NORMAL_THRESHOLD, HEALTH_WARNING_THRESHOLD, MODELS_DIR
from .preprocessing import validate_feature_mapping, validate_feature_vector


def classify_health(rul: float) -> str:
    """Rule-based health classification from predicted RUL (in cycles)."""
    if rul > HEALTH_NORMAL_THRESHOLD:
        return "Normal"
    if rul >= HEALTH_WARNING_THRESHOLD:
        return "Warning"
    return "Critical"


def load_artifacts(model_dir: Path | str = MODELS_DIR) -> Optional[dict[str, Any]]:
    model_dir = Path(model_dir)
    model_path = model_dir / "random_forest_rul.joblib"
    features_path = model_dir / "feature_names.json"
    metadata_path = model_dir / "model_metadata.json"
    if not model_path.exists() or not features_path.exists():
        return None
    model = joblib.load(model_path)
    with open(features_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    return {"model": model, "feature_names": feature_names, "metadata": metadata}


def prepare_input(
    features: Any, feature_names: list[str]
) -> np.ndarray:
    """Convert dict/list input into a validated 2D array in feature order."""
    if isinstance(features, dict):
        values = validate_feature_mapping(features, feature_names)
    elif isinstance(features, (list, tuple)):
        values = validate_feature_vector(list(features), feature_names)
    else:
        raise ValueError("Features must be an object or array.")
    return pd.DataFrame([values], columns=feature_names)


def predict_rul(artifacts: dict[str, Any], features: Any) -> float:
    if artifacts is None:
        raise RuntimeError("Model artifacts are not loaded.")
    model = artifacts["model"]
    feature_names = artifacts["feature_names"]
    X = prepare_input(features, feature_names)
    pred = float(model.predict(X)[0])
    # RUL cannot be negative.
    if pred < 0:
        pred = 0.0
    return pred


def get_feature_importances(artifacts: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
    if not artifacts:
        return []
    model = artifacts.get("model")
    names = artifacts.get("feature_names", [])
    if model is None or not hasattr(model, "feature_importances_"):
        return []
    importances = getattr(model, "feature_importances_")
    pairs = sorted(zip(names, importances), key=lambda p: float(p[1]), reverse=True)
    return [{"feature": str(f), "importance": float(i)} for f, i in pairs]


def predict_rul_range(artifacts: dict[str, Any], features: Any) -> tuple[float, float, float]:
    rul = predict_rul(artifacts, features)
    model = artifacts["model"]
    feature_names = artifacts["feature_names"]
    X = prepare_input(features, feature_names)
    low = rul
    high = rul
    if hasattr(model, "estimators_"):
        try:
            tree_preds = np.asarray([float(est.predict(X)[0]) for est in model.estimators_], dtype=float)
            low = float(np.percentile(tree_preds, 10))
            high = float(np.percentile(tree_preds, 90))
        except Exception:
            low = rul
            high = rul
    else:
        metadata = artifacts.get("metadata", {})
        residual_low = metadata.get("residual_q10", 0.0)
        residual_high = metadata.get("residual_q90", 0.0)
        low = rul + float(residual_low)
        high = rul + float(residual_high)
    low = max(0.0, min(low, high))
    high = max(low, high)
    if rul < low:
        low = rul
    if rul > high:
        high = rul
    return rul, low, high


def build_explanation(predicted_rul: float, health: str) -> str:
    return (
        f"Estimated remaining useful life is {predicted_rul:.1f} cycles. "
        f"Health status '{health}' is derived from configured RUL thresholds "
        f"(Normal > 30, Warning 10-30, Critical < 10 cycles)."
    )

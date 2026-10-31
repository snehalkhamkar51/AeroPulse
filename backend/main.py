"""FastAPI application for AeroPulse."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Ensure src/ is importable when running from project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

import io

import pandas as pd

from aeropulse import config, predict
from aeropulse.config import MODELS_DIR
from aeropulse.data_loader import load_cmapss, DatasetError
from aeropulse.history_features import build_history_features, history_feature_names, SELECTED_SENSORS
from backend.schemas import (
    EvaluationDataResponse,
    EvaluationHistogramBin,
    EvaluationPoint,
    HealthResponse,
    HistoryCyclePrediction,
    HistoryEnginePrediction,
    ModelInfoResponse,
    PredictHistoryResponse,
    PredictRequest,
    PredictResponse,
    SampleEnginesResponse,
)

app = FastAPI(title="AeroPulse API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Loaded once at startup; not retrained per request.
_ARTIFACTS = None


@app.on_event("startup")
def _load_model() -> None:
    global _ARTIFACTS
    _ARTIFACTS = predict.load_artifacts(MODELS_DIR)


def _artifacts():
    global _ARTIFACTS
    if _ARTIFACTS is None:
        _ARTIFACTS = predict.load_artifacts(MODELS_DIR)
    return _ARTIFACTS

@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    artifacts = _artifacts()
    ready = artifacts is not None
    return HealthResponse(
        status="ok",
        model_ready=ready,
        model_path=None,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    artifacts = _artifacts()
    if artifacts is None:
        return ModelInfoResponse(status="not_trained")
    metadata = artifacts.get("metadata", {})
    model = artifacts.get("model")
    model_params = {}
    if model is not None:
        for attr in ("n_estimators", "max_depth", "min_samples_split", "min_samples_leaf", "max_features", "random_state", "bootstrap"):
            if hasattr(model, attr):
                val = getattr(model, attr)
                if hasattr(val, "item") and callable(getattr(val, "item", None)):
                    val = val.item()
                model_params[attr] = val
    return ModelInfoResponse(
        model_type=metadata.get("model_type"),
        subset=metadata.get("subset"),
        trained_at=metadata.get("trained_at"),
        n_features=metadata.get("n_features"),
        feature_names=artifacts.get("feature_names", []),
        rul_clip=metadata.get("rul_clip"),
        validation_metrics=metadata.get("val_metrics"),
        test_metrics=metadata.get("test_metrics"),
        feature_importances=predict.get_feature_importances(artifacts),
        model_params=model_params,
        status="ready",
    )


@app.post("/predict", response_model=PredictResponse)
def predict_endpoint(request: PredictRequest) -> PredictResponse:
    artifacts = _artifacts()
    if artifacts is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not trained or artifacts are unavailable.",
        )
    feature_names = artifacts["feature_names"]
    try:
        if request.features is not None:
            features_input = request.features
        else:
            features_input = request.feature_vector
        rul, rul_low, rul_high = predict.predict_rul_range(artifacts, features_input)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Prediction failed.")

    health_status = predict.classify_health(rul)
    explanation = predict.build_explanation(rul, health_status)
    contributions = predict.get_feature_importances(artifacts)
    # Attach input values to contributions for explainability page.
    if request.features is not None:
        fmap = request.features
        for c in contributions:
            c["value"] = fmap.get(c["feature"])
    else:
        vec = request.feature_vector or []
        for idx, c in enumerate(contributions):
            # map by feature name order
            pass
        # Build value map from feature order.
        value_map = dict(zip(feature_names, vec)) if vec else {}
        for c in contributions:
            c["value"] = value_map.get(c["feature"])

    metadata = artifacts.get("metadata", {}) if artifacts else {}
    model_version = metadata.get("model_version") or f"{metadata.get('subset', 'FD001')}-{metadata.get('trained_at', 'unknown')}"
    thresholds = {"normal_gt": 30.0, "warning_min": 10.0, "warning_max": 30.0, "critical_lt": 10.0}
    return PredictResponse(
        predicted_rul=round(rul, 4),
        health_status=health_status,
        explanation=explanation,
        unit_id=request.unit_id,
        features_used=len(feature_names),
        feature_contributions=contributions,
        model_ready=True,
        thresholds=thresholds,
        model_version=model_version,
        rul_low=round(rul_low, 4),
        rul_high=round(rul_high, 4),
    )


_SAMPLE_CACHE = None


def _load_sample_engines():
    global _SAMPLE_CACHE
    if _SAMPLE_CACHE is not None:
        return _SAMPLE_CACHE

    artifacts = _artifacts()
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not trained or artifacts are unavailable.")

    feature_names = artifacts["feature_names"]
    model = artifacts["model"]
    try:
        train_df, test_df, rul_series = load_cmapss(config.DATA_DIR, "FD001")
    except DatasetError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    feature_stats = {}
    for f in feature_names:
        col = train_df[f] if f != "cycle" else train_df["time_cycles"]
        feature_stats[f] = {
            "median": float(col.median()),
            "min": float(col.min()),
            "max": float(col.max()),
        }

    units = sorted(rul_series.index.tolist(), key=lambda u: float(rul_series[u]))
    n = len(units)
    picks = sorted(
        set(
            [
                units[0],
                units[1],
                units[max(0, n // 4)],
                units[max(0, n // 2)],
                units[max(0, 3 * n // 4)],
                units[-2],
                units[-1],
            ]
        )
    )

    def category(final_rul: float) -> str:
        if final_rul > 50:
            return "healthy"
        if final_rul >= 20:
            return "mid-life"
        return "near-failure"

    engines = []
    for unit in picks:
        group = test_df[test_df["unit_number"] == unit].sort_values("time_cycles")
        if group.empty:
            continue
        last_cycle = int(group["time_cycles"].max())
        official_final = float(rul_series[unit])
        capped_final = min(official_final, 125.0)
        cycles = [int(x) for x in group["time_cycles"].tolist()]
        feature_frame = group[[f for f in feature_names if f != "cycle"]].copy()
        feature_frame["cycle"] = group["time_cycles"].astype(float).to_numpy()
        feature_frame = feature_frame[feature_names]
        features = feature_frame.astype(float).values.tolist()
        true_rul = [min(official_final + (last_cycle - cycle), 125.0) for cycle in cycles]
        predicted_rul = [float(value) for value in model.predict(feature_frame)]
        engines.append(
            {
                "id": int(unit),
                "category": category(capped_final),
                "final_rul": capped_final,
                "cycles": cycles,
                "feature_names": feature_names,
                "features": features,
                "true_rul": true_rul,
                "predicted_rul": predicted_rul,
            }
        )

    _SAMPLE_CACHE = {
        "feature_names": feature_names,
        "feature_stats": feature_stats,
        "engines": engines,
    }
    return _SAMPLE_CACHE


@app.get("/sample-engines", response_model=SampleEnginesResponse)
def sample_engines():
    return _load_sample_engines()


@app.get("/sample-engines/{engine_id}")
def sample_engine(engine_id: int):
    cache = _load_sample_engines()
    for engine in cache["engines"]:
        if engine["id"] == engine_id:
            return engine

_EVAL_CACHE = None


def _load_evaluation_data():
    global _EVAL_CACHE
    if _EVAL_CACHE is not None:
        return _EVAL_CACHE

    artifacts = _artifacts()
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not trained or artifacts are unavailable.")

    feature_names = artifacts["feature_names"]
    model = artifacts["model"]
    try:
        train_df, test_df, rul_series = load_cmapss(config.DATA_DIR, "FD001")
    except DatasetError as exc:
        raise HTTPException(status_code=503, detail=str(exc))

    last_rows = (
        test_df.sort_values(["unit_number", "time_cycles"])
        .groupby("unit_number", as_index=False)
        .tail(1)
        .reset_index(drop=True)
    )
    X_test = last_rows[feature_names]
    preds = model.predict(X_test)
    units = sorted(last_rows["unit_number"].tolist())
    true = np.asarray([rul_series[u] for u in units], dtype=float)
    order = np.argsort(last_rows["unit_number"].to_numpy())
    pred_sorted = np.asarray(preds)[order]
    true_sorted = true  # true already sorted by units? rul_series indexed sorted; yes

    points = []
    errors = []
    for i, unit in enumerate(units):
        err = float(pred_sorted[i] - true_sorted[i])
        errors.append(err)
        points.append(
            {
                "engine_id": int(unit),
                "true_rul": float(true_sorted[i]),
                "predicted_rul": float(pred_sorted[i]),
                "error": err,
            }
        )

    counts, edges = np.histogram(errors, bins=10)
    histogram = []
    for i in range(len(counts)):
        histogram.append(
            {
                "bin_start": float(edges[i]),
                "bin_end": float(edges[i + 1]),
                "count": int(counts[i]),
            }
        )

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))

    _EVAL_CACHE = {
        "points": points,
        "histogram": histogram,
        "mae": mae,
        "rmse": rmse,
    }
    return _EVAL_CACHE


@app.get("/evaluation-data", response_model=EvaluationDataResponse)
def evaluation_data():
    return _load_evaluation_data()

    raise HTTPException(status_code=404, detail=f"Sample engine {engine_id} not found.")

@app.post("/predict-history", response_model=PredictHistoryResponse)
async def predict_history(file: UploadFile = File(...)):
    artifacts = _artifacts()
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not trained or artifacts are unavailable.")

    feature_names = artifacts["feature_names"]
    model = artifacts["model"]
    metadata = artifacts.get("metadata", {})
    raw_features = [f for f in feature_names if f != "cycle"]
    model_version = metadata.get("model_version") or f"{metadata.get('subset', 'FD001')}-{metadata.get('trained_at', 'unknown')}"

    try:
        content = await file.read()
        text = content.decode("utf-8-sig")
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="CSV file is empty.")

    required = ["cycle", *raw_features]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise HTTPException(status_code=422, detail=f"Missing required columns: {missing}")

    if "engine_id" not in df.columns:
        df["engine_id"] = "engine_1"

    # Validate finite numeric values.
    for col in required:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise HTTPException(status_code=422, detail=f"Column '{col}' must be numeric.")
        if df[col].isna().any():
            raise HTTPException(status_code=422, detail=f"Column '{col}' contains missing values.")
        if not np.isfinite(df[col].to_numpy(dtype=float)).all():
            raise HTTPException(status_code=422, detail=f"Column '{col}' contains non-finite values.")

    if not np.allclose(df["cycle"], df["cycle"].round()):
        raise HTTPException(status_code=422, detail="Cycle values must be integers.")
    if (df["cycle"] <= 0).any():
        raise HTTPException(status_code=422, detail="Cycle values must be positive.")

    engines_out = []
    for engine_id, group in df.groupby("engine_id", sort=True):
        group = group.sort_values("cycle").reset_index(drop=True)
        if len(group) < 5:
            raise HTTPException(status_code=422, detail=f"Engine {engine_id} has fewer than 5 rows.")
        if group["cycle"].duplicated().any():
            raise HTTPException(status_code=422, detail=f"Engine {engine_id} has duplicate cycle values.")

        frame = pd.DataFrame(
            {
                "unit_number": 0,
                "time_cycles": group["cycle"].astype(int).to_numpy(),
**{f: group[f].astype(float).to_numpy() for f in raw_features},
            }
        )
        built = build_history_features(frame, sensors=raw_features)
        if built.empty:
            raise HTTPException(status_code=422, detail=f"Engine {engine_id} does not have enough valid history.")

        # Rolling features are validated/built for history uploads. The active model
        # uses current-cycle sensor values plus cycle number.
        X = built[feature_names]
        preds = model.predict(X)
        cycle_list = [int(x) for x in built["time_cycles"].tolist()]
        rul_by_cycle = [{"cycle": c, "predicted_rul": float(p)} for c, p in zip(cycle_list, preds)]

        # Range for latest prediction.
        latest_features = X.iloc[[-1]]
        latest_pred = float(preds[-1])
        low = latest_pred
        high = latest_pred
        if hasattr(model, "estimators_"):
            try:
                tree_preds = np.asarray([float(est.predict(latest_features)[0]) for est in model.estimators_], dtype=float)
                low = float(np.percentile(tree_preds, 10))
                high = float(np.percentile(tree_preds, 90))
            except Exception:
                low = latest_pred
                high = latest_pred
        low = max(0.0, min(low, high))
        high = max(low, high)
        if latest_pred < low:
            low = latest_pred
        if latest_pred > high:
            high = latest_pred

        latest_cycle = int(cycle_list[-1])
        status = predict.classify_health(latest_pred)
        engines_out.append(
            {
                "engine_id": str(engine_id),
                "latest_cycle": latest_cycle,
                "predicted_rul": round(latest_pred, 4),
                "health_status": status,
                "rul_low": round(low, 4),
                "rul_high": round(high, 4),
                "rul_by_cycle": rul_by_cycle,
            }
        )

    return {
        "engines": engines_out,
        "model_version": model_version,
        "feature_names": feature_names,
    }

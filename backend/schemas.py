"""Pydantic schemas for the AeroPulse backend API."""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class HealthResponse(BaseModel):
    status: str
    model_ready: bool
    model_path: Optional[str] = None


class FeatureStats(BaseModel):
    median: float
    min: float
    max: float


class SampleEngine(BaseModel):
    id: int
    category: str
    final_rul: float
    cycles: list[int]
    feature_names: list[str]
    features: list[list[float]]
    true_rul: list[float]
    predicted_rul: list[float] = []


class SampleEnginesResponse(BaseModel):
    feature_names: list[str]
    feature_stats: dict[str, FeatureStats]
    engines: list[SampleEngine]


class ModelInfoResponse(BaseModel):
    model_type: Optional[str] = None
    subset: Optional[str] = None
    trained_at: Optional[str] = None
    n_features: Optional[int] = None
    feature_names: list[str] = Field(default_factory=list)
    rul_clip: Optional[int] = None
    validation_metrics: Optional[dict[str, Any]] = None
    test_metrics: Optional[dict[str, Any]] = None
    feature_importances: list[dict[str, Any]] = Field(default_factory=list)
    model_params: dict[str, Any] = Field(default_factory=dict)
    status: str

class EvaluationPoint(BaseModel):
    engine_id: int
    true_rul: float
    predicted_rul: float
    error: float


class EvaluationHistogramBin(BaseModel):
    bin_start: float
    bin_end: float
    count: int


class EvaluationDataResponse(BaseModel):
    points: list[EvaluationPoint]
    histogram: list[EvaluationHistogramBin]
    mae: float
    rmse: float


class PredictRequest(BaseModel):
    model_config = {"extra": "forbid"}
    features: Optional[dict[str, float]] = None
    feature_vector: Optional[list[float]] = None
    unit_id: Optional[int] = None

    @model_validator(mode="after")
    def _require_input(self):
        if self.features is None and self.feature_vector is None:
            raise ValueError("Either 'features' or 'feature_vector' must be provided.")
        return self

    @field_validator("features")
    @classmethod
    def _check_features_finite(cls, v):
        if v is None:
            return v
        for k, val in v.items():
            if val is None:
                raise ValueError(f"Missing value for '{k}'.")
            try:
                fv = float(val)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Non-numeric value for '{k}'.") from exc
            if fv != fv or fv in (float("inf"), float("-inf")):
                raise ValueError(f"Feature '{k}' must be finite.")
        return v

    @field_validator("feature_vector")
    @classmethod
    def _check_vector_finite(cls, v):
        if v is None:
            return v
        for val in v:
            try:
                fv = float(val)
            except (TypeError, ValueError) as exc:
                raise ValueError("Non-numeric value in feature_vector.") from exc
            if fv != fv or fv in (float("inf"), float("-inf")):
                raise ValueError("feature_vector values must be finite.")
        return v


class PredictResponse(BaseModel):
    predicted_rul: float
    health_status: str
    explanation: str
    unit_id: Optional[int] = None
    features_used: int
    feature_contributions: list[dict[str, Any]] = Field(default_factory=list)
    model_ready: bool = True
    thresholds: dict[str, float] = Field(default_factory=dict)
    model_version: str = ""
    rul_low: float = 0.0
    rul_high: float = 0.0



class HistoryCyclePrediction(BaseModel):
    cycle: int
    predicted_rul: float


class HistoryEnginePrediction(BaseModel):
    engine_id: str
    latest_cycle: int
    predicted_rul: float
    health_status: str
    rul_low: float
    rul_high: float
    rul_by_cycle: list[HistoryCyclePrediction]


class PredictHistoryResponse(BaseModel):
    engines: list[HistoryEnginePrediction]
    model_version: str = ""
    feature_names: list[str] = Field(default_factory=list)

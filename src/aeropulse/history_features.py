"""History-aware feature engineering for AeroPulse RUL models."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

HISTORY_WINDOW = 30
MIN_HISTORY = 5
BASELINE_CYCLES = 5

# Compact sensors retained from the current-cycle model.
SELECTED_SENSORS = [
    "sensor_11",
    "sensor_9",
    "sensor_4",
    "sensor_12",
    "sensor_7",
    "sensor_15",
    "sensor_21",
]


def _linear_slope(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return 0.0
    x = np.arange(values.size, dtype=float)
    x = x - x.mean()
    y = values - values.mean()
    denominator = float(np.dot(x, x))
    if denominator == 0.0:
        return 0.0
    return float(np.dot(x, y) / denominator)


def history_feature_names(sensors: Iterable[str] = SELECTED_SENSORS) -> list[str]:
    names: list[str] = []
    for sensor in sensors:
        names.extend(
            [
                sensor,
                f"{sensor}_roll_mean",
                f"{sensor}_roll_std",
                f"{sensor}_slope",
                f"{sensor}_delta_baseline",
            ]
        )
    names.append("cycle")
    return names


def build_history_features(
    df: pd.DataFrame,
    sensors: Iterable[str] = SELECTED_SENSORS,
    window: int = HISTORY_WINDOW,
    min_periods: int = MIN_HISTORY,
) -> pd.DataFrame:
    """Build rolling history features without using future observations.

    Rows with fewer than ``min_periods`` observations for an engine are dropped.
    """
    sensors = list(sensors)
    required = ["unit_number", "time_cycles", *sensors]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work = df.copy()
    work["unit_number"] = work["unit_number"].astype(int)
    work["time_cycles"] = work["time_cycles"].astype(int)
    if "RUL" not in work.columns:
        max_cycles = work.groupby("unit_number")["time_cycles"].transform("max")
        work["RUL"] = max_cycles - work["time_cycles"]
    work = work.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)

    feature_frames: list[pd.DataFrame] = []
    for unit, group in work.groupby("unit_number", sort=True):
        group = group.sort_values("time_cycles").copy()
        for sensor in sensors:
            rolling = group[sensor].rolling(window=window, min_periods=min_periods)
            group[f"{sensor}_roll_mean"] = rolling.mean()
            group[f"{sensor}_roll_std"] = rolling.std(ddof=1)
            group[f"{sensor}_slope"] = rolling.apply(_linear_slope, raw=True)
            baseline_cycles = min(BASELINE_CYCLES, len(group))
            baseline = float(group[sensor].iloc[:baseline_cycles].mean())
            group[f"{sensor}_delta_baseline"] = group[sensor] - baseline
        group["cycle"] = group["time_cycles"].astype(float)
        feature_frames.append(group)

    if not feature_frames:
        return pd.DataFrame(columns=["unit_number", "time_cycles", *history_feature_names(sensors)])

    result = pd.concat(feature_frames, ignore_index=True)
    result = result.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)

    feature_cols = history_feature_names(sensors)
    # Keep only rows where the rolling features are available.
    result = result.dropna(subset=feature_cols).reset_index(drop=True)
    if "RUL" in df.columns:
        # Preserve RUL aligned by unit and cycle.
        rul_lookup = df.set_index(["unit_number", "time_cycles"])["RUL"]
        result["RUL"] = result.set_index(["unit_number", "time_cycles"]).index.map(rul_lookup)
    return result


def history_feature_frame(df: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Return only metadata plus engineered feature columns."""
    built = build_history_features(df, **kwargs)
    keep = ["unit_number", "time_cycles", *history_feature_names(kwargs.get("sensors", SELECTED_SENSORS))]
    if "RUL" in built.columns:
        keep.append("RUL")
    return built[keep].copy()

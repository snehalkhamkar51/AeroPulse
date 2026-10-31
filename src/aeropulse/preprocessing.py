"""Feature selection and preprocessing helpers (consistent train/inference)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from .config import HEALTH_NORMAL_THRESHOLD, HEALTH_WARNING_THRESHOLD


EXCLUDE_COLUMNS = {"unit_number", "time_cycles", "RUL"}


def feature_candidate_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in EXCLUDE_COLUMNS]


def drop_constant_features(
    df: pd.DataFrame, feature_cols: list[str], min_std: float = 1e-12
) -> tuple[list[str], list[str]]:
    stds = df[feature_cols].std(ddof=0)
    keep = [c for c in feature_cols if float(stds[c]) > min_std]
    dropped = [c for c in feature_cols if c not in keep]
    return keep, dropped


def make_training_arrays(
    train_df: pd.DataFrame, rul_clip: Optional[int] = None, include_cycle: bool = False
) -> tuple[pd.DataFrame, pd.Series, pd.Series, list[str], list[str]]:
    """Prepare X, y, groups, feature_names and dropped columns from raw train df."""
    from .data_loader import add_train_rul

    df = add_train_rul(train_df, clip=rul_clip)
    candidates = feature_candidate_columns(df)
    feature_names, dropped = drop_constant_features(df, candidates)
    X = df[feature_names].copy()
    if include_cycle:
        X["cycle"] = df["time_cycles"].astype(float)
        feature_names = feature_names + ["cycle"]
    y = df["RUL"].copy()
    groups = df["unit_number"].copy()
    return X, y, groups, feature_names, dropped


def select_features(df: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    out = df[feature_names].copy()
    # Validate finite.
    arr = out.to_numpy(dtype=float)
    if not np.isfinite(arr).all():
        raise ValueError("Input contains NaN or infinite values.")
    return out


def split_by_engine(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, set, set]:
    """Split data by engine id (no engine leakage)."""
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    engines = sorted(groups.unique().tolist())
    rng = np.random.default_rng(seed)
    shuffled = list(engines)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * test_size)))
    val_engines = set(shuffled[:n_val])
    train_engines = set(shuffled[n_val:])

    train_mask = ~groups.isin(val_engines)
    val_mask = groups.isin(val_engines)

    X_train = X.loc[train_mask].reset_index(drop=True)
    X_val = X.loc[val_mask].reset_index(drop=True)
    y_train = y.loc[train_mask].reset_index(drop=True)
    y_val = y.loc[val_mask].reset_index(drop=True)

    # Verify no leakage.
    assert train_engines.isdisjoint(val_engines)
    assert set(groups.loc[train_mask].unique()).isdisjoint(set(groups.loc[val_mask].unique()))
    return X_train, X_val, y_train, y_val, train_engines, val_engines


def validate_feature_vector(values, feature_names: list[str]) -> list[float]:
    """Validate a positional feature vector."""
    if not isinstance(values, (list, tuple)):
        raise ValueError("Feature vector must be a list.")
    if len(values) != len(feature_names):
        raise ValueError(
            f"Expected {len(feature_names)} features, got {len(values)}."
        )
    out: list[float] = []
    for v in values:
        try:
            fv = float(v)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric feature value: {v!r}") from exc
        if not np.isfinite(fv):
            raise ValueError("Feature values must be finite (no NaN/inf).")
        out.append(fv)
    return out


def validate_feature_mapping(mapping: dict, feature_names: list[str]) -> list[float]:
    """Validate a feature-name -> value mapping and return ordered values."""
    if not isinstance(mapping, dict):
        raise ValueError("Features must be provided as an object mapping.")
    missing = [f for f in feature_names if f not in mapping]
    if missing:
        raise ValueError(f"Missing required features: {missing}")
    out: list[float] = []
    for name in feature_names:
        try:
            fv = float(mapping[name])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Non-numeric value for '{name}': {mapping[name]!r}") from exc
        if not np.isfinite(fv):
            raise ValueError(f"Feature '{name}' must be finite (no NaN/inf).")
        out.append(fv)
    return out

"""Loading and validation helpers for the NASA C-MAPSS dataset."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import COLUMN_NAMES, FD_SUBSETS


class DatasetError(Exception):
    """Raised when the dataset is missing or incompatible."""


def _find_file(data_dir: Path, prefix: str, subset: str) -> Optional[Path]:
    data_dir = Path(data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        return None
    expected = f"{prefix}_{subset}.txt".lower()
    # Search recursively so nested extracted archives are supported.
    candidates = [p for p in data_dir.rglob("*") if p.is_file()]
    for path in candidates:
        if path.name.lower() == expected:
            return path
    for path in candidates:
        name = path.name.lower()
        if prefix.lower() in name and subset.lower() in name:
            return path
    return None


def detect_subset_files(data_dir: Path, subset: str = "FD001") -> dict[str, Path]:
    data_dir = Path(data_dir)
    result: dict[str, Path] = {}
    missing: list[str] = []
    for split, prefix in (("train", "train"), ("test", "test"), ("rul", "RUL")):
        path = _find_file(data_dir, prefix, subset)
        if path is None:
            missing.append(f"{prefix}_{subset}.txt")
        else:
            result[split] = path
    if missing:
        raise DatasetError(
            "Missing required dataset files in "
            f"{data_dir}: {', '.join(missing)}. "
            "Place the extracted NASA C-MAPSS files into the data/ folder."
        )
    return result


def _read_cmapss(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(
            path, sep=r"\s+", header=None, names=COLUMN_NAMES, engine="python"
        )
    except Exception as exc:
        raise DatasetError(f"Could not read {path}: {exc}") from exc

    if df.shape[1] != len(COLUMN_NAMES):
        raise DatasetError(f"{path.name} has {df.shape[1]} columns; expected {len(COLUMN_NAMES)}.")
    if df.empty:
        raise DatasetError(f"{path.name} is empty.")

    for col in COLUMN_NAMES:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise DatasetError(f"Column '{col}' in {path.name} is not numeric.")
    if df.isna().any().any():
        raise DatasetError(f"{path.name} contains missing (NaN) values.")
    if not np.isfinite(df.select_dtypes(include=[np.number]).to_numpy()).all():
        raise DatasetError(f"{path.name} contains non-finite values.")

    if not (df["unit_number"] > 0).all():
        raise DatasetError(f"{path.name} contains non-positive unit numbers.")
    if not (df["time_cycles"] > 0).all():
        raise DatasetError(f"{path.name} contains non-positive cycle numbers.")
    if not np.allclose(df["unit_number"], df["unit_number"].round()):
        raise DatasetError(f"{path.name} unit_number column is not integer-valued.")
    if not np.allclose(df["time_cycles"], df["time_cycles"].round()):
        raise DatasetError(f"{path.name} time_cycles column is not integer-valued.")

    df["unit_number"] = df["unit_number"].astype(int)
    df["time_cycles"] = df["time_cycles"].astype(int)

    # Duplicate observations.
    if df.duplicated(subset=["unit_number", "time_cycles"]).any():
        dup_count = int(df.duplicated(subset=["unit_number", "time_cycles"]).sum())
        raise DatasetError(f"{path.name} contains {dup_count} duplicate (unit, cycle) observations.")

    # Cycle sequences should be contiguous starting at 1 for each engine.
    for unit, g in df.groupby("unit_number"):
        cycles = sorted(g["time_cycles"].tolist())
        expected = list(range(1, len(cycles) + 1))
        if cycles != expected:
            raise DatasetError(
                f"{path.name} engine {unit} has a non-contiguous or non-1-based cycle sequence."
            )

    return df

def load_cmapss(data_dir, subset="FD001"):
    data_dir = Path(data_dir)
    if subset not in FD_SUBSETS:
        raise DatasetError(f"Unknown subset '{subset}'. Expected one of {FD_SUBSETS}.")

    files = detect_subset_files(data_dir, subset)
    train_df = _read_cmapss(files["train"]).sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)
    test_df = _read_cmapss(files["test"]).sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)

    try:
        rul_values = pd.read_csv(files["rul"], header=None, engine="python")
    except Exception as exc:
        raise DatasetError(f"Could not read {files['rul']}: {exc}") from exc
    rul_values = rul_values.dropna(axis=1, how="all")
    if rul_values.shape[1] != 1:
        rul_values = pd.DataFrame(rul_values.values.reshape(-1, 1))
    rul_flat = rul_values.iloc[:, 0].astype(float).to_numpy().ravel()
    if not np.isfinite(rul_flat).all():
        raise DatasetError("RUL file contains non-finite values.")
    if (rul_flat < 0).any():
        raise DatasetError("RUL file contains negative values.")

    test_units = sorted(test_df["unit_number"].unique().tolist())
    if len(test_units) != len(rul_flat):
        raise DatasetError(
            f"RUL file has {len(rul_flat)} entries but test set has {len(test_units)} engines."
        )
    rul_series = pd.Series(
        rul_flat, index=pd.Index(test_units, name="unit_number"), name="RUL"
    )

    return train_df, test_df, rul_series


def add_train_rul(train_df, clip=None):
    if "unit_number" not in train_df.columns or "time_cycles" not in train_df.columns:
        raise DatasetError("Training data missing unit_number/time_cycles columns.")
    df = train_df.copy()
    max_cycles = df.groupby("unit_number")["time_cycles"].max()
    df["RUL"] = df["unit_number"].map(max_cycles) - df["time_cycles"]
    if (df["RUL"] < 0).any():
        raise DatasetError("Derived negative RUL values; check cycle data.")
    if clip is not None:
        df["RUL"] = df["RUL"].clip(upper=clip)
    return df


def summarize_dataset(train_df, test_df, rul_series):
    return {
        "train_rows": int(len(train_df)),
        "train_engines": int(train_df["unit_number"].nunique()),
        "test_rows": int(len(test_df)),
        "test_engines": int(test_df["unit_number"].nunique()),
        "rul_entries": int(len(rul_series)),
    }

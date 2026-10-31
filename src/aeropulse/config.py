"""AeroPulse configuration constants.

Central place for paths, seeds, thresholds and dataset column definitions.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root: src/aeropulse -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = Path(os.getenv("AEROPULSE_DATA_DIR", str(PROJECT_ROOT / "data")))
MODELS_DIR = Path(os.getenv("AEROPULSE_MODELS_DIR", str(PROJECT_ROOT / "models")))
REPORTS_DIR = Path(os.getenv("AEROPULSE_REPORTS_DIR", str(PROJECT_ROOT / "reports")))

RANDOM_SEED = 42

# RUL clipping threshold (cycles). Set to None to disable clipping.
RUL_CLIP: int | None = 125

# Engine health classification thresholds (in cycles).
HEALTH_NORMAL_THRESHOLD = 30  # RUL > 30  -> Normal
HEALTH_WARNING_THRESHOLD = 10  # 10 <= RUL <= 30 -> Warning; RUL < 10 -> Critical

# Standard C-MAPSS column layout (26 columns).
COLUMN_NAMES = (
    ["unit_number", "time_cycles", "op_setting_1", "op_setting_2", "op_setting_3"]
    + [f"sensor_{i}" for i in range(1, 22)]
)

FD_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]

# Artifact filenames.
MODEL_FILENAME = "random_forest_rul.joblib"
FEATURES_FILENAME = "feature_names.json"
METADATA_FILENAME = "model_metadata.json"

# CORS origins for the planned frontend.
CORS_ORIGINS = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:4173",
    "http://127.0.0.1:4173",
]

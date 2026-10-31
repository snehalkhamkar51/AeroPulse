"""Training entry point for the RUL regression model."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor

from .config import (
    MODELS_DIR,
    RANDOM_SEED,
    REPORTS_DIR,
    RUL_CLIP,
    MODEL_FILENAME,
    FEATURES_FILENAME,
    METADATA_FILENAME,
)
from .data_loader import load_cmapss, summarize_dataset
from .evaluation import regression_metrics
from .preprocessing import make_training_arrays, split_by_engine


def train_model(
    data_dir,
    subset="FD001",
    test_size=0.2,
    seed=RANDOM_SEED,
    rul_clip=RUL_CLIP,
    model_dir=MODELS_DIR,
    report_dir=REPORTS_DIR,
    features=None,
):
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    report_dir = Path(report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train_df, test_df, rul_series = load_cmapss(data_dir, subset)
    dataset_summary = summarize_dataset(train_df, test_df, rul_series)

    include_cycle = bool(features and "cycle" in features)
    X_all, y, groups, all_feature_names, dropped_constant = make_training_arrays(train_df, rul_clip=rul_clip, include_cycle=include_cycle)
    if features:
        requested = [str(f).strip() for f in features]
        missing = [f for f in requested if f not in X_all.columns]
        if missing:
            raise ValueError(f"Requested features are not available: {missing}")
        X = X_all[requested].copy()
        feature_names = requested
        dropped = [f for f in all_feature_names if f not in requested] + dropped_constant
    else:
        X = X_all
        feature_names = all_feature_names
        dropped = dropped_constant

    X_train, X_val, y_train, y_val, train_engines, val_engines = split_by_engine(
        X, y, groups, test_size=test_size, seed=seed
    )

    model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=-1)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_metrics = regression_metrics(y_val, val_pred)

    # Official test evaluation: last observed cycle per test engine.
    test_metrics = None
    try:
        last_rows = (
            test_df.sort_values(["unit_number", "time_cycles"])
            .groupby("unit_number", as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        X_test = last_rows.copy()
        if "cycle" in feature_names and "cycle" not in X_test.columns:
            X_test["cycle"] = last_rows["time_cycles"].astype(float)
        X_test = X_test[feature_names]
        test_pred = model.predict(X_test)
        units_sorted = sorted(last_rows["unit_number"].tolist())
        true_rul = np.asarray([rul_series[u] for u in units_sorted], dtype=float)
        order = np.argsort(last_rows["unit_number"].to_numpy())
        test_pred_sorted = np.asarray(test_pred)[order]
        test_metrics = regression_metrics(true_rul, test_pred_sorted)
    except Exception as exc:
        test_metrics = {"error": str(exc)}

    # Save artifacts.
    joblib.dump(model, model_dir / "random_forest_rul.joblib")
    with open(model_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2)

    importances = []
    if hasattr(model, "feature_importances_"):
        importances = sorted(
            zip(feature_names, model.feature_importances_.tolist()),
            key=lambda p: p[1],
            reverse=True,
        )
        importances = [{"feature": f, "importance": float(i)} for f, i in importances]

    metadata = {
        "model_type": type(model).__name__,
        "subset": subset,
        "random_seed": seed,
        "rul_clip": rul_clip,
        "test_size": test_size,
        "n_train_samples": int(len(X_train)),
        "n_val_samples": int(len(X_val)),
        "n_features": int(len(feature_names)),
        "dropped_features": dropped,
        "feature_names": feature_names,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "dataset_summary": dataset_summary,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "feature_importances": importances,
    }
    with open(model_dir / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    report = {
        "project": "AeroPulse",
        "objective": "Estimate Remaining Useful Life (RUL) of turbofan engines.",
        "subset": subset,
        "dataset_summary": dataset_summary,
        "rul_labeling": {
            "strategy": "max_cycle_per_engine - current_cycle",
            "clipping_threshold": rul_clip,
            "clipping_note": (
                "RUL values above the threshold are clipped to reduce dominance of early-life samples."
                if rul_clip is not None
                else "No clipping applied."
            ),
        },
        "split": {
            "method": "engine-id grouped split",
            "test_size": test_size,
            "seed": seed,
            "train_engines": sorted(train_engines),
            "val_engines": sorted(val_engines),
        },
        "model": {"type": "RandomForestRegressor", "n_estimators": 200, "random_state": seed},
        "features": {"used": feature_names, "dropped_constant": dropped},
        "metrics": {"validation": val_metrics, "test_official": test_metrics},
        "feature_importances": importances,
        "limitations": [
            "Predictions are estimates, not certified maintenance decisions.",
            "Feature importances are global impurity-based measures, not causal effects.",
        ],
    }
    with open(report_dir / "training_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    md_lines = [
        "# AeroPulse Training Report",
        "",
        f"- Subset: {subset}",
        f"- Train samples: {len(X_train)}  Val samples: {len(X_val)}",
        f"- Features: {len(feature_names)}",
        f"- Validation MAE: {val_metrics['mae']:.4f}",
        f"- Validation RMSE: {val_metrics['rmse']:.4f}",
    ]
    if isinstance(test_metrics, dict) and "mae" in test_metrics:
        md_lines.append(f"- Official test MAE: {test_metrics['mae']:.4f}")
        md_lines.append(f"- Official test RMSE: {test_metrics['rmse']:.4f}")
    md_lines.append("")
    md_lines.append("## Dropped (constant) features")
    md_lines.append(", ".join(dropped) if dropped else "None")
    (report_dir / "training_report.md").write_text("\n".join(md_lines), encoding="utf-8")

    return {
        "model": model,
        "feature_names": feature_names,
        "dropped": dropped,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "metadata": metadata,
    }


def main():
    parser = argparse.ArgumentParser(description="Train AeroPulse RUL model.")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--subset", type=str, default="FD001")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--rul-clip", type=int, default=RUL_CLIP, help="RUL clipping threshold; use -1 to disable.")
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--features", type=str, default=None, help="Comma-separated feature list to train on.")
    args = parser.parse_args()

    from .config import DATA_DIR, MODELS_DIR as MD, REPORTS_DIR as RD
    from .feature_list import CURRENT_FEATURES

    data_dir = Path(args.data_dir) if args.data_dir else DATA_DIR
    model_dir = Path(args.model_dir) if args.model_dir else MD
    report_dir = Path(args.report_dir) if args.report_dir else RD
    rul_clip = None if (args.rul_clip is not None and args.rul_clip < 0) else args.rul_clip
    if args.features:
        features = [x.strip() for x in args.features.split(",") if x.strip()]
    else:
        features = list(CURRENT_FEATURES)

    result = train_model(
        data_dir=data_dir,
        subset=args.subset,
        test_size=args.test_size,
        seed=args.seed,
        rul_clip=rul_clip,
        model_dir=model_dir,
        report_dir=report_dir,
        features=features,
    )
    print("Training complete.")
    print("Validation metrics:", result["val_metrics"])
    print("Test metrics:", result["test_metrics"])


if __name__ == "__main__":
    main()


from pathlib import Path
import sys
import json
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
sys.path.insert(0, r"E:\AeroPulse\src")
from aeropulse.config import RANDOM_SEED, RUL_CLIP
from aeropulse.data_loader import load_cmapss
from aeropulse.preprocessing import make_training_arrays, split_by_engine
from aeropulse.evaluation import regression_metrics

data_dir = Path(r"E:\AeroPulse\data")
compact_features = ["sensor_11", "sensor_9", "sensor_4", "sensor_12", "sensor_7", "sensor_15", "sensor_21"]

train_df, test_df, rul_series = load_cmapss(data_dir, "FD001")
X, y, groups, all_features, dropped = make_training_arrays(train_df, rul_clip=RUL_CLIP)
missing = [f for f in compact_features if f not in X.columns]
if missing:
    raise SystemExit(f"Missing compact features: {missing}")

Xc = X[compact_features].copy()
Xtr, Xva, ytr, yva, tr_eng, va_eng = split_by_engine(Xc, y, groups, test_size=0.2, seed=RANDOM_SEED)

model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
model.fit(Xtr, ytr)
val_pred = model.predict(Xva)
val_metrics = regression_metrics(yva, val_pred)

last_rows = test_df.sort_values(["unit_number", "time_cycles"]).groupby("unit_number", as_index=False).tail(1).reset_index(drop=True)
X_test = last_rows[compact_features]
test_pred = model.predict(X_test)
units_sorted = sorted(last_rows["unit_number"].tolist())
true_rul = np.asarray([rul_series[u] for u in units_sorted], dtype=float)
order = np.argsort(last_rows["unit_number"].to_numpy())
test_pred_sorted = np.asarray(test_pred)[order]
test_metrics = regression_metrics(true_rul, test_pred_sorted)

print("COMPACT FEATURES:", compact_features)
print("Validation:", val_metrics)
print("Test:", test_metrics)

out_dir = Path(r"E:\AeroPulse\_compact_candidate")
out_dir.mkdir(parents=True, exist_ok=True)
joblib.dump(model, out_dir / "random_forest_rul_compact.joblib")
(out_dir / "feature_names_compact.json").write_text(json.dumps(compact_features, indent=2), encoding="utf-8")
meta = {
    "model_type": type(model).__name__,
    "subset": "FD001",
    "random_seed": RANDOM_SEED,
    "rul_clip": RUL_CLIP,
    "test_size": 0.2,
    "n_train_samples": int(len(Xtr)),
    "n_val_samples": int(len(Xva)),
    "n_features": len(compact_features),
    "dropped_features": [f for f in all_features if f not in compact_features],
    "feature_names": compact_features,
    "val_metrics": val_metrics,
    "test_metrics": test_metrics,
}
(out_dir / "model_metadata_compact.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
print("Saved candidate to", out_dir)

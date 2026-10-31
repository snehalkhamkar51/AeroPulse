from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import joblib
sys.path.insert(0, r"E:\AeroPulse\src")
from aeropulse.config import RANDOM_SEED, RUL_CLIP
from aeropulse.data_loader import load_cmapss
from aeropulse.evaluation import regression_metrics, nasa_score
from aeropulse.history_features import build_history_features, history_feature_names
from aeropulse.preprocessing import make_training_arrays, split_by_engine

data_dir = Path(r"E:\AeroPulse\data")
train_df, test_df, rul_series = load_cmapss(data_dir, "FD001")

# Current compact model
Xc, yc, gc, all_features, _ = make_training_arrays(train_df, rul_clip=RUL_CLIP)
compact_features = json.load(open(r"E:\AeroPulse\models\feature_names.json"))
Xc = Xc[compact_features]
_, _, _, _, train_engines, val_engines = split_by_engine(Xc, yc, gc, test_size=0.2, seed=RANDOM_SEED)
compact_model = joblib.load(Path(r"E:\AeroPulse\models/random_forest_rul.joblib"))

val_all_mask = gc.isin(val_engines)
Xva_all = Xc.loc[val_all_mask].reset_index(drop=True)
yva_all = yc.loc[val_all_mask].reset_index(drop=True)
last_val_indices = train_df.loc[val_all_mask, "unit_number"].groupby(train_df.loc[val_all_mask, "unit_number"]).tail(1).index if False else train_df[train_df["unit_number"].isin(val_engines)].groupby("unit_number")["time_cycles"].idxmax()
Xva_last = Xc.loc[last_val_indices].reset_index(drop=True)
yva_last = yc.loc[last_val_indices].reset_index(drop=True)

compact_val_all_pred = compact_model.predict(Xva_all)
compact_val_last_pred = compact_model.predict(Xva_last)

# Test last cycle, capped and uncapped true RUL
last_test = test_df.sort_values(["unit_number", "time_cycles"]).groupby("unit_number", as_index=False).tail(1).reset_index(drop=True)
X_test_compact = last_test[compact_features]
compact_test_pred = compact_model.predict(X_test_compact)
units = sorted(last_test["unit_number"].tolist())
true_test_uncapped = np.asarray([rul_series[u] for u in units], dtype=float)
true_test_capped = np.minimum(true_test_uncapped, RUL_CLIP)
order = np.argsort(last_test["unit_number"].to_numpy())
compact_test_pred_sorted = np.asarray(compact_test_pred)[order]

print("CURRENT COMPACT MODEL")
print("validation all-row:", regression_metrics(yva_all, compact_val_all_pred))
print("validation last-cycle:", regression_metrics(yva_last, compact_val_last_pred))
print("test last-cycle uncapped true:", regression_metrics(true_test_uncapped, compact_test_pred_sorted))
print("test last-cycle capped true:", regression_metrics(true_test_capped, compact_test_pred_sorted))
print("test NASA uncapped:", nasa_score(true_test_uncapped, compact_test_pred_sorted))
print("test NASA capped:", nasa_score(true_test_capped, compact_test_pred_sorted))

# History model
train_h = build_history_features(train_df)
test_h = build_history_features(test_df)
hist_features = history_feature_names()
Xh = train_h[hist_features]
yh = train_h["RUL"].astype(float)
gh = train_h["unit_number"]
_, _, _, _, h_train_engines, h_val_engines = split_by_engine(Xh, yh, gh, test_size=0.2, seed=RANDOM_SEED)
history_model = joblib.load(Path(r"E:\AeroPulse\models/random_forest_rul_history.joblib"))

h_val_all_mask = gh.isin(h_val_engines)
Xhva_all = Xh.loc[h_val_all_mask].reset_index(drop=True)
yhva_all = yh.loc[h_val_all_mask].reset_index(drop=True)
h_last_val_indices = train_h[train_h["unit_number"].isin(h_val_engines)].groupby("unit_number")["time_cycles"].idxmax()
Xhva_last = Xh.loc[h_last_val_indices].reset_index(drop=True)
yhva_last = yh.loc[h_last_val_indices].reset_index(drop=True)

hist_val_all_pred = history_model.predict(Xhva_all)
hist_val_last_pred = history_model.predict(Xhva_last)

last_test_h = test_h.sort_values(["unit_number", "time_cycles"]).groupby("unit_number", as_index=False).tail(1).reset_index(drop=True)
Xh_test = last_test_h[hist_features]
hist_test_pred = history_model.predict(Xh_test)
hist_units = sorted(last_test_h["unit_number"].tolist())
hist_true_uncapped = np.asarray([rul_series[u] for u in hist_units], dtype=float)
hist_true_capped = np.minimum(hist_true_uncapped, RUL_CLIP)
hist_order = np.argsort(last_test_h["unit_number"].to_numpy())
hist_test_pred_sorted = np.asarray(hist_test_pred)[hist_order]

print("\nHISTORY MODEL")
print("validation all-row:", regression_metrics(yhva_all, hist_val_all_pred))
print("validation last-cycle:", regression_metrics(yhva_last, hist_val_last_pred))
print("test last-cycle uncapped true:", regression_metrics(hist_true_uncapped, hist_test_pred_sorted))
print("test last-cycle capped true:", regression_metrics(hist_true_capped, hist_test_pred_sorted))
print("test NASA uncapped:", nasa_score(hist_true_uncapped, hist_test_pred_sorted))
print("test NASA capped:", nasa_score(hist_true_capped, hist_test_pred_sorted))

# Feature builder unit check
tiny = pd.DataFrame({
    "unit_number": [1, 1, 1, 1, 1],
    "time_cycles": [1, 2, 3, 4, 5],
    "sensor_11": [10.0, 11.0, 12.0, 13.0, 14.0],
})
from aeropulse.history_features import build_history_features
built = build_history_features(tiny, sensors=["sensor_11"], window=5, min_periods=5)
print("\nFEATURE BUILDER TINY ENGINE")
print(built[["time_cycles", "sensor_11", "sensor_11_roll_mean", "sensor_11_slope", "sensor_11_delta_baseline"]].to_string(index=False))
print("expected rolling mean last:", sum([10,11,12,13,14])/5)
print("expected slope:", ( (0-2)*(-2) + (1-2)*(-1)+ (2-2)*0 + (3-2)*1 + (4-2)*2 ) / (0**2+1**2+2**2+3**2+4**2 - 5*2**2) if False else "manual")
PY
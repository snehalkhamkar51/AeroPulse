from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
sys.path.insert(0, r"E:\AeroPulse\src")
from aeropulse.config import RANDOM_SEED, RUL_CLIP
from aeropulse.data_loader import load_cmapss
from aeropulse.evaluation import regression_metrics
from aeropulse.history_features import build_history_features, SELECTED_SENSORS
from aeropulse.preprocessing import split_by_engine

data_dir = Path(r"E:\AeroPulse\data")
train_df, test_df, rul_series = load_cmapss(data_dir, "FD001")
sensors = SELECTED_SENSORS
train_h = build_history_features(train_df, sensors=sensors)
test_h = build_history_features(test_df, sensors=sensors)
# keep only current + rolling means + cycle
keep = [s for s in sensors] + [f"{s}_roll_mean" for s in sensors] + ["cycle"]
X = train_h[keep]
y = train_h["RUL"].astype(float)
groups = train_h["unit_number"]
Xtr, Xva, ytr, yva, _, _ = split_by_engine(X, y, groups, test_size=0.2, seed=RANDOM_SEED)
m = RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1)
m.fit(Xtr, ytr)
val_pred = m.predict(Xva)
last = test_h.sort_values(["unit_number","time_cycles"]).groupby("unit_number", as_index=False).tail(1).reset_index(drop=True)
Xt = last[keep]
pt = m.predict(Xt)
units = sorted(last["unit_number"].tolist())
true = np.asarray([rul_series[u] for u in units], dtype=float)
order = np.argsort(last["unit_number"].to_numpy())
pt = np.asarray(pt)[order]
print("features", len(keep))
print("val", regression_metrics(yva, val_pred))
print("test", regression_metrics(true, pt))
PY
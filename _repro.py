import json, io, pandas as pd, sys
sys.path.insert(0, "src")
from aeropulse.history_features import build_history_features
from aeropulse import predict
import backend.main as main
art = main._artifacts()
feature_names = art["feature_names"]
raw_features = [f for f in feature_names if f != "cycle"]
samples = __import__("urllib.request", fromlist=["x"]).urlopen("http://127.0.0.1:8000/sample-engines").read() if False else None
# use a sample engine from direct data instead
from aeropulse.data_loader import load_cmapss
from pathlib import Path
train_df, test_df, rul = load_cmapss(Path("data"), "FD001")
eng = 1
g = test_df[test_df.unit_number == eng].sort_values("time_cycles").head(8)
frame = pd.DataFrame({
    "unit_number": 0,
    "time_cycles": g["time_cycles"].astype(int).to_numpy(),
    **{f: g[f].astype(float).to_numpy() for f in raw_features},
})
built = build_history_features(frame, sensors=raw_features)
print("built columns has cycle:", "cycle" in built.columns)
print("feature_names:", feature_names)
print("missing:", [f for f in feature_names if f not in built.columns])
print("built shape:", built.shape)

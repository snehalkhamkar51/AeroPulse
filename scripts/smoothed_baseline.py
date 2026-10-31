from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
sys.path.insert(0, r"E:\AeroPulse\src")
from aeropulse.config import RANDOM_SEED, RUL_CLIP
from aeropulse.data_loader import load_cmapss, add_train_rul
from aeropulse.evaluation import regression_metrics, nasa_score
from aeropulse.preprocessing import split_by_engine

DATA = Path("data/archive/CMaps")
train_df, test_df, rul_series = load_cmapss(DATA, "FD001")
train_df = train_df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)
test_df = test_df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True)

nonconst_sensors = [c for c in train_df.columns if c.startswith("sensor_") and train_df[c].std(ddof=0) > 1e-12]
print("Non-constant sensors:", len(nonconst_sensors))
print(nonconst_sensors)

TOP_SENSORS = ["sensor_11", "sensor_9", "sensor_4"]


def make_features(df, mode):
    df = df.sort_values(["unit_number", "time_cycles"]).reset_index(drop=True).copy()
    out = pd.DataFrame(index=df.index)
    out["unit_number"] = df["unit_number"].astype(int)
    out["time_cycles"] = df["time_cycles"].astype(int)
    if mode == "raw":
        for s in nonconst_sensors:
            out[s] = df[s].astype(float)
    elif mode == "smooth":
        for s in nonconst_sensors:
            out[s] = df[s].astype(float)
            g = df.groupby("unit_number")[s]
            out[f"{s}_rm5"] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
            out[f"{s}_rm10"] = g.transform(lambda x: x.rolling(10, min_periods=1).mean())
    elif mode == "smooth_slope":
        for s in nonconst_sensors:
            out[s] = df[s].astype(float)
            g = df.groupby("unit_number")[s]
            out[f"{s}_rm5"] = g.transform(lambda x: x.rolling(5, min_periods=1).mean())
            out[f"{s}_rm10"] = g.transform(lambda x: x.rolling(10, min_periods=1).mean())
        for s in TOP_SENSORS:
            g = df.groupby("unit_number")[s]
            out[f"{s}_slope10"] = g.transform(lambda x: x.rolling(10, min_periods=1).apply(lambda v: np.polyfit(range(len(v)), v, 1)[0] if len(v) > 1 else 0.0, raw=True))
    out["cycle"] = df["time_cycles"].astype(float)
    if "RUL" in df.columns:
        out["RUL"] = df["RUL"].astype(float)
    else:
        max_cycles = df.groupby("unit_number")["time_cycles"].transform("max")
        out["RUL"] = (max_cycles - df["time_cycles"]).clip(upper=RUL_CLIP).astype(float)
    return out

sample = train_df[train_df["unit_number"] == 1].head(12).copy()
sample["RUL"] = (sample["time_cycles"].max() - sample["time_cycles"]).clip(upper=RUL_CLIP)
feat_sample = make_features(sample, "smooth")
display_cols = ["time_cycles", "sensor_11", "sensor_11_rm5", "sensor_11_rm10", "sensor_9", "sensor_9_rm5", "sensor_9_rm10", "cycle"]
print("First 12 rows, engine 1 raw vs smoothed:")
print(feat_sample[display_cols].to_string(index=False))

modes = ["raw", "smooth", "smooth_slope"]
model_factories = {
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
    "extra_trees": lambda: ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_SEED, n_jobs=-1),
    "hist_gradient_boosting": lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=RANDOM_SEED),
}

for mode in modes:
    train_h = make_features(train_df, mode)
    test_h = make_features(test_df, mode)
    if mode == "raw":
        feature_cols = nonconst_sensors + ["cycle"]
    elif mode == "smooth":
        feature_cols = [f"{s}_rm5" for s in nonconst_sensors] + [f"{s}_rm10" for s in nonconst_sensors] + ["cycle"]
    else:
        feature_cols = [f"{s}_rm5" for s in nonconst_sensors] + [f"{s}_rm10" for s in nonconst_sensors] + [f"{s}_slope10" for s in TOP_SENSORS] + ["cycle"]
    X = train_h[feature_cols]
    y = train_h["RUL"]
    groups = train_h["unit_number"]
    X_train, X_val, y_train, y_val, _, _ = split_by_engine(X, y, groups, test_size=0.2, seed=RANDOM_SEED)

    engines = sorted(groups.unique().tolist())
    rng = np.random.default_rng(RANDOM_SEED)
    shuffled = list(engines)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * 0.2)))
    val_engine_set = set(shuffled[:n_val])
    val_mask = groups.isin(val_engine_set)
    X_val_all = X.loc[val_mask].reset_index(drop=True)
    y_val_all = y.loc[val_mask].reset_index(drop=True)
    last_val_indices = train_h.loc[val_mask].groupby("unit_number")["time_cycles"].idxmax()
    X_val_last = X.loc[last_val_indices].reset_index(drop=True)
    y_val_last = y.loc[last_val_indices].reset_index(drop=True)

    last_test = test_h.sort_values(["unit_number", "time_cycles"]).groupby("unit_number", as_index=False).tail(1).reset_index(drop=True)
    X_test = last_test[feature_cols]
    true_test = np.asarray([rul_series[u] for u in sorted(last_test["unit_number"].tolist())], dtype=float)
    true_test_capped = np.minimum(true_test, RUL_CLIP)
    test_order = np.argsort(last_test["unit_number"].to_numpy())

    print(f"\n=== FEATURE SET: {mode} ({len(feature_cols)} features) ===")
    for model_name, factory in model_factories.items():
        model = factory()
        model.fit(X_train, y_train)
        val_all_pred = model.predict(X_val_all)
        val_last_pred = model.predict(X_val_last)
        test_pred = np.asarray(model.predict(X_test))[test_order]
        print(f"{model_name}:")
        print("  val all-row  MAE/RMSE:", regression_metrics(y_val_all, val_all_pred))
        print("  val last-cycle MAE/RMSE:", regression_metrics(y_val_last, val_last_pred))
        print("  test last-cycle uncapped MAE/RMSE:", regression_metrics(true_test, test_pred))
        print("  test last-cycle capped   MAE/RMSE:", regression_metrics(true_test_capped, test_pred))
        print("  test NASA uncapped:", round(nasa_score(true_test, test_pred), 2))

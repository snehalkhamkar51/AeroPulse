from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold
sys.path.insert(0, r"E:\AeroPulse\src")
from aeropulse.config import RANDOM_SEED, RUL_CLIP, DATA_DIR
from aeropulse.data_loader import load_cmapss
from aeropulse.evaluation import regression_metrics, nasa_score
from aeropulse.history_features import build_history_features, history_feature_names
from aeropulse.preprocessing import split_by_engine

data_dir = Path(r"E:\AeroPulse\data")
train_df, test_df, rul_series = load_cmapss(data_dir, "FD001")
train_h = build_history_features(train_df)
test_h = build_history_features(test_df)

feature_names = history_feature_names()
X = train_h[feature_names]
y = train_h["RUL"].astype(float)
groups = train_h["unit_number"]

X_train, X_val, y_train, y_val, train_engines, val_engines = split_by_engine(X, y, groups, test_size=0.2, seed=RANDOM_SEED)

cv_mask = train_h["unit_number"].isin(train_engines)
X_cv = train_h.loc[cv_mask, feature_names].reset_index(drop=True)
y_cv = train_h.loc[cv_mask, "RUL"].astype(float).reset_index(drop=True)
g_cv = train_h.loc[cv_mask, "unit_number"].reset_index(drop=True)

last_test = test_h.sort_values(["unit_number", "time_cycles"]).groupby("unit_number", as_index=False).tail(1).reset_index(drop=True)
X_test = last_test[feature_names]
true_test = np.asarray([rul_series[u] for u in sorted(last_test["unit_number"].tolist())], dtype=float)
test_order = np.argsort(last_test["unit_number"].to_numpy())


def cv_rmse(factory, params):
    gkf = GroupKFold(n_splits=3)
    rmses = []
    for tr_idx, va_idx in gkf.split(X_cv, y_cv, g_cv):
        model = factory(random_state=RANDOM_SEED, **params)
        model.fit(X_cv.iloc[tr_idx], y_cv.iloc[tr_idx])
        pred = model.predict(X_cv.iloc[va_idx])
        rmses.append(float(np.sqrt(np.mean((pred - y_cv.iloc[va_idx].to_numpy()) ** 2))))
    return float(np.mean(rmses))


def fit_eval(factory, params):
    model = factory(random_state=RANDOM_SEED, **params)
    model.fit(X_train, y_train)
    val_pred = model.predict(X_val)
    test_pred = model.predict(X_test)
    test_pred_sorted = np.asarray(test_pred)[test_order]
    return model, {
        "val": regression_metrics(y_val, val_pred),
        "test": regression_metrics(true_test, test_pred_sorted),
        "val_nasa": nasa_score(y_val, val_pred),
        "test_nasa": nasa_score(true_test, test_pred_sorted),
    }


results = []

rf_grid = [
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 2},
]
rf_params = min(rf_grid, key=lambda p: cv_rmse(RandomForestRegressor, p))
rf_model, rf_metrics = fit_eval(RandomForestRegressor, rf_params)
results.append(("random_forest", rf_params, rf_model, rf_metrics))

et_grid = [
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 2},
]
et_params = min(et_grid, key=lambda p: cv_rmse(ExtraTreesRegressor, p))
et_model, et_metrics = fit_eval(ExtraTreesRegressor, et_params)
results.append(("extra_trees", et_params, et_model, et_metrics))

hgb_grid = [
    {"max_iter": 300, "learning_rate": 0.05, "max_depth": None, "l2_regularization": 0.0},
    {"max_iter": 300, "learning_rate": 0.10, "max_depth": 6, "l2_regularization": 1.0},
    {"max_iter": 300, "learning_rate": 0.05, "max_depth": 4, "l2_regularization": 1.0},
]
hgb_params = min(hgb_grid, key=lambda p: cv_rmse(HistGradientBoostingRegressor, p))
hgb_model, hgb_metrics = fit_eval(HistGradientBoostingRegressor, hgb_params)
results.append(("hist_gradient_boosting", hgb_params, hgb_model, hgb_metrics))

for name, params, model, metrics in results:
    print(name, params)
    print("  validation:", metrics["val"], "NASA:", round(metrics["val_nasa"], 2))
    print("  test:", metrics["test"], "NASA:", round(metrics["test_nasa"], 2))

best = min(results, key=lambda item: item[3]["val"]["rmse"])
print("BEST BY VALIDATION RMSE:", best[0], best[1])
print("BEST METRICS:", best[3])
PY
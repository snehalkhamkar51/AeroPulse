from pathlib import Path
import sys
sys.path.insert(0, r"E:\AeroPulse\src")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from aeropulse.data_loader import load_cmapss
from aeropulse.preprocessing import make_training_arrays, split_by_engine

data_dir = Path(r"E:\AeroPulse\data")
train_df, test_df, rul_series = load_cmapss(data_dir, "FD001")
X, y, groups, features, dropped = make_training_arrays(train_df, rul_clip=125)
Xtr, Xva, ytr, yva, tr_eng, va_eng = split_by_engine(X, y, groups, test_size=0.2, seed=42)

model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
model.fit(Xtr, ytr)

imp = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
print("IMPUITY IMPORTANCES")
print(imp.to_string())

perm = permutation_importance(model, Xva, yva, n_repeats=10, random_state=42, n_jobs=-1, scoring="neg_mean_absolute_error")
perm_series = pd.Series(perm.importances_mean, index=features).sort_values(ascending=False)
print("\nPERMUTATION IMPORTANCE (validation MAE decrease)")
print(perm_series.to_string())

print("\nNEAR-CONSTANT (train std / abs median)")
for f in features:
    col = X[f]
    med = col.median()
    std = col.std(ddof=0)
    rng = col.max() - col.min()
    print(f"{f}: median={med:.4f} std={std:.4f} range={rng:.4f} rel={ (std/abs(med) if med != 0 else float('inf')):.4f}")

corr = X.corr().abs()
print("\nHIGHLY CORRELATED PAIRS |r| > 0.95")
pairs = []
for i in range(len(features)):
    for j in range(i+1, len(features)):
        r = corr.iloc[i,j]
        if r > 0.95:
            pairs.append((features[i], features[j], r))
for a,b,r in sorted(pairs, key=lambda x:-x[2]):
    print(f"{a} ~ {b}: {r:.4f}")
print("num pairs:", len(pairs))

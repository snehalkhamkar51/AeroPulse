import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from aeropulse import train as train_mod
from aeropulse.config import COLUMN_NAMES


def _make_dataset(tmp_path, n_engines=4, cycles=8):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    rows = []
    for u in range(1, n_engines + 1):
        for c in range(1, cycles + 1):
            rows.append([u, c] + [float(rng.normal()) for _ in range(24)])
    df = pd.DataFrame(rows, columns=COLUMN_NAMES)
    df.to_csv(tmp_path / "train_FD001.txt", sep=" ", header=False, index=False)

    trows = []
    for u in range(1, n_engines + 1):
        for c in range(1, cycles):
            trows.append([u, c] + [float(rng.normal()) for _ in range(24)])
    tdf = pd.DataFrame(trows, columns=COLUMN_NAMES)
    tdf.to_csv(tmp_path / "test_FD001.txt", sep=" ", header=False, index=False)

    (tmp_path / "RUL_FD001.txt").write_text("\n".join(str(5 + i) for i in range(n_engines)) + "\n")
    return tmp_path


def test_train_and_save_load(tmp_path):
    data_dir = _make_dataset(tmp_path / "data")
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    result = train_mod.train_model(
        data_dir=data_dir,
        subset="FD001",
        test_size=0.25,
        seed=7,
        rul_clip=5,
        model_dir=model_dir,
        report_dir=report_dir,
    )
    assert (model_dir / "random_forest_rul.joblib").exists()
    assert (model_dir / "feature_names.json").exists()
    assert (model_dir / "model_metadata.json").exists()
    assert (report_dir / "training_report.json").exists()

    from aeropulse.predict import load_artifacts, predict_rul, classify_health
    art = load_artifacts(model_dir)
    assert art is not None
    assert len(art["feature_names"]) > 0

    # predict using first feature vector from training data
    import pandas as pd
    train_df = pd.read_csv(data_dir / "train_FD001.txt", sep=r"\s+", header=None, names=COLUMN_NAMES, engine="python")
    feats = art["feature_names"]
    row = train_df[feats].iloc[0].tolist()
    pred = predict_rul(art, row)
    assert isinstance(pred, float)
    assert pred >= 0
    assert classify_health(pred) in {"Normal", "Warning", "Critical"}


def test_train_missing_data_raises(tmp_path):
    with pytest.raises(Exception):
        train_mod.train_model(data_dir=tmp_path / "nope", subset="FD001", model_dir=tmp_path / "m", report_dir=tmp_path / "r")

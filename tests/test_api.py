import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from aeropulse import predict, train as train_mod
from aeropulse.config import COLUMN_NAMES


def _make_dataset(tmp_path, n_engines=4, cycles=8):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(2)
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


@pytest.fixture
def client():
    import backend.main as main
    # reset
    main._ARTIFACTS = None
    from fastapi.testclient import TestClient
    return TestClient(main.app)


def test_health_not_trained(client, monkeypatch):
    import backend.main as main
    monkeypatch.setattr("backend.main.predict.load_artifacts", lambda d: None)
    main._ARTIFACTS = None
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert data["model_ready"] is False


def test_model_info_not_trained(client, monkeypatch):
    import backend.main as main
    monkeypatch.setattr("backend.main.predict.load_artifacts", lambda d: None)
    main._ARTIFACTS = None
    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["status"] == "not_trained"


def test_predict_not_trained(client, monkeypatch):
    import backend.main as main
    monkeypatch.setattr("backend.main.predict.load_artifacts", lambda d: None)
    main._ARTIFACTS = None
    r = client.post("/predict", json={"feature_vector": [1.0] * 24})
    # model not ready -> 503 (but feature count mismatch may 422 first? model check first)
    assert r.status_code == 503


def test_predict_trained(client, tmp_path, monkeypatch):
    import backend.main as main
    data_dir = _make_dataset(tmp_path / "data")
    model_dir = tmp_path / "models"
    report_dir = tmp_path / "reports"
    train_mod.train_model(data_dir=data_dir, subset="FD001", test_size=0.25, seed=3, rul_clip=5, model_dir=model_dir, report_dir=report_dir)
    art = predict.load_artifacts(model_dir)
    main._ARTIFACTS = art
    monkeypatch.setattr("backend.main.predict.load_artifacts", lambda d: art)
    monkeypatch.setattr("backend.main.MODELS_DIR", model_dir)

    r = client.get("/health")
    assert r.json()["model_ready"] is True

    r = client.get("/model-info")
    assert r.status_code == 200
    assert r.json()["status"] == "ready"
    assert len(r.json()["feature_names"]) == len(art["feature_names"])

    # valid predict via features dict
    feats = art["feature_names"]
    fv = {f: 0.5 for f in feats}
    r = client.post("/predict", json={"features": fv})
    assert r.status_code == 200
    data = r.json()
    assert "predicted_rul" in data
    assert data["health_status"] in {"Normal", "Warning", "Critical"}
    assert data["features_used"] == len(feats)
    assert isinstance(data["feature_contributions"], list)

    # invalid: missing features
    r = client.post("/predict", json={"features": {}})
    assert r.status_code in (422,)

    # nan
    bad = {f: 0.5 for f in feats}
    bad[feats[0]] = None
    r = client.post("/predict", json={"features": bad})
    assert r.status_code == 422

    # feature_vector wrong length
    r = client.post("/predict", json={"feature_vector": [1.0, 2.0]})
    assert r.status_code == 422

def _csv_bytes(feature_names, rows):
    import io
    buf = io.StringIO()
    buf.write("engine_id,cycle," + ",".join(feature_names) + "\n")
    for engine_id, cycle, values in rows:
        buf.write(f"{engine_id},{cycle}," + ",".join(str(v) for v in values) + "\n")
    return buf.getvalue().encode()


def test_predict_history_valid(client):
    info = client.get("/model-info").json()
    feats = info["feature_names"]
    samples = client.get("/sample-engines").json()
    eng = samples["engines"][0]
    rows = [(1, c, row) for c, row in zip(eng["cycles"][:8], eng["features"][:8])]
    files = {"file": ("history.csv", _csv_bytes(feats, rows), "text/csv")}
    r = client.post("/predict-history", files=files)
    assert r.status_code == 200
    data = r.json()
    assert len(data["engines"]) == 1
    assert len(data["engines"][0]["rul_by_cycle"]) >= 1
    assert "predicted_rul" in data["engines"][0]


def test_predict_history_missing_columns(client):
    info = client.get("/model-info").json()
    feats = info["feature_names"]
    bad = "engine_id," + ",".join(feats) + "\n1," + ",".join(str(x) for x in [0.5] * len(feats)) + "\n"
    files = {"file": ("bad.csv", bad.encode(), "text/csv")}
    r = client.post("/predict-history", files=files)
    assert r.status_code == 422


def test_predict_history_too_few_rows(client):
    info = client.get("/model-info").json()
    feats = info["feature_names"]
    rows = [(1, 1, [0.5] * len(feats))]
    files = {"file": ("short.csv", _csv_bytes(feats, rows), "text/csv")}
    r = client.post("/predict-history", files=files)
    assert r.status_code == 422


def test_predict_history_nan(client):
    info = client.get("/model-info").json()
    feats = info["feature_names"]
    rows = [(1, c, [float("nan")] + [0.5] * (len(feats) - 1)) for c in range(1, 6)]
    files = {"file": ("nan.csv", _csv_bytes(feats, rows), "text/csv")}
    r = client.post("/predict-history", files=files)
    assert r.status_code == 422


def test_predict_history_multi_engine(client):
    info = client.get("/model-info").json()
    feats = info["feature_names"]
    samples = client.get("/sample-engines").json()
    e1 = samples["engines"][0]
    e2 = samples["engines"][1]
    rows = []
    for c, row in zip(e1["cycles"][:6], e1["features"][:6]):
        rows.append((e1["id"], c, row))
    for c, row in zip(e2["cycles"][:6], e2["features"][:6]):
        rows.append((e2["id"], c, row))
    files = {"file": ("multi.csv", _csv_bytes(feats, rows), "text/csv")}
    r = client.post("/predict-history", files=files)
    assert r.status_code == 200
    data = r.json()
    assert len(data["engines"]) == 2

# AeroPulse — AI-Powered Engine Degradation & RUL Countdown Tracker

AeroPulse estimates an aircraft (turbofan) engine's **Remaining Useful Life (RUL)** in
operational cycles, tracks degradation, and communicates an estimated engine health
status (Normal / Warning / Critical).

> Predictions are statistical estimates based on the NASA C-MAPSS dataset. They are
> prototype indicators, **not** certified aircraft maintenance or aviation safety decisions.

## Planned application pages

1. **Landing Page** — project introduction, title, animated 3D aircraft engine.
2. **Engine Inputs Page** — operating settings and sensor measurements.
3. **Results Dashboard** — predicted RUL, countdown presentation, health classification.
4. **About/Project Page** — dataset, methodology, limitations, degradation story.

> Phase 1 (this prompt) implements the ML foundation and backend. The frontend, 3D
> animation and final integration are delivered in Prompts 2 and 3.

## Architecture and technologies

- Python, pandas, NumPy
- scikit-learn (`RandomForestRegressor`)
- FastAPI + Pydantic + Uvicorn
- joblib for model artifacts
- pytest for tests

Project layout:

```
AeroPulse/
├── data/            # place extracted NASA C-MAPSS files here (.gitkeep tracked)
├── models/          # trained artifacts (.gitkeep tracked)
├── reports/         # training reports (.gitkeep tracked)
├── src/aeropulse/   # ML package
├── backend/         # FastAPI app
├── tests/           # pytest suite
├── requirements.txt
├── .env.example
└── README.md
```

## Dataset

Source: NASA C-MAPSS turbofan engine degradation dataset (Kaggle mirror). Prefer the
**FD001** subset. Place extracted files into `data/`:

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

The loader also supports FD002/FD003/FD004. Raw dataset files are gitignored.

Each row has 26 space-separated columns: unit number, time (cycles), 3 operating
settings, 21 sensor measurements.

## Windows PowerShell setup

```powershell
cd E:\AeroPulse
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks script execution, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Training and evaluation

Default data directory is `./data`. To train (FD001):

```powershell
cd E:\AeroPulse
$env:PYTHONPATH = "src"
python -m aeropulse.train --data-dir .\data --subset FD001
```

Or with a custom location (e.g. the extracted archive):

```powershell
python -m aeropulse.train --data-dir E:\archive\CMaps --subset FD001
```

Options: `--test-size`, `--seed`, `--rul-clip` (use `-1` to disable clipping), `--model-dir`, `--report-dir`.

Artifacts are written to `models/` and a report to `reports/training_report.json` (+ `.md`).

RUL labeling strategy: `RUL = (engine max cycle) - (current cycle)`, optionally
clipped at 125 cycles (configurable) to reduce dominance of early-life samples.

Validation split is performed **by engine id** to prevent leakage.

## Tests

```powershell
cd E:\AeroPulse
$env:PYTHONPATH = "src;."
pytest
```

Tests using the real dataset skip cleanly when files are absent. Synthetic fixtures
verify code behavior; they are never presented as NASA results.

## Backend startup and API usage

```powershell
cd E:\AeroPulse
$env:PYTHONPATH = "src;."
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Endpoints:

- `GET /health` — API availability and model readiness.
- `GET /model-info` — model type, features, training status, metrics, importances.
- `POST /predict` — validate input, run the trained model, return RUL + health + explanation.

Example request body (all trained feature names are required; see
`models/feature_names.json` for the exact list and order):

```json
{
  "features": {
    "op_setting_1": 0.0,
    "op_setting_2": 0.0,
    "sensor_2": 642.0
  }
}
```

A positional `feature_vector` aligned with `feature_names` is also accepted.

Health classification thresholds (cycles):

- Normal: RUL > 30
- Warning: 10 <= RUL <= 30
- Critical: RUL < 10

## Input/output contract

- Input: complete feature vector (operating settings + sensor measurements actually
  used by the trained model). No training labels or future RUL values are requested.
- Output: `predicted_rul` (cycles), `health_status`, `explanation`,
  `feature_contributions` (global importances with input values), model readiness.

Missing/malformed/NaN/inf inputs are rejected with HTTP 422.

## Explainability

Global feature importance (Random Forest impurity-based) is saved in metadata and
exposed via `/model-info` and `/predict`. These are **global** measures, not causal
effects and not per-prediction contributions. Per-prediction SHAP-style
contributions are not computed in Phase 1.

## Limitations

- Predictions are estimates; not certified maintenance decisions.
- Constant sensors are dropped; subset-specific.
- No frontend/3D yet (Prompts 2–3).

## What is implemented vs pending

Implemented (Phase 1): data loading/validation, RUL labeling, engine-wise split,
preprocessing, RF training, save/load, evaluation, health classification,
FastAPI endpoints, explainability data, tests, docs.

Pending (Prompts 2–3): React/TS/Vite frontend, 3D engine animation, full
integration, end-to-end testing, final documentation.


## Verified evaluation metrics (FD001)

- Validation MAE: 12.89 cycles
- Validation RMSE: 18.30 cycles
- Official test MAE: 13.12 cycles
- Official test RMSE: 18.01 cycles

These are genuine results from the trained RandomForestRegressor; see `reports/training_report.json`.


## Frontend (React + Vite)

```powershell
cd E:\AeroPulse\frontend
npm install
copy .env.example .env
npm run dev
```

The dev server runs on `http://localhost:5173` and connects to the backend via
`VITE_API_BASE_URL` (default `http://127.0.0.1:8000`).

Build for production:
```powershell
npm run build
npm run preview
```

Run frontend tests:
```powershell
npm run test
```


## Helper scripts

`scripts/audit_features.py` and `scripts/train_compact.py` are optional maintenance scripts. They are not part of the application runtime.

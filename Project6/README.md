# Bike Rental Demand — Reproducible MLOps Pipeline

Predict **hourly bike-rental demand** from weather, calendar, and recent-usage
signals, and serve those predictions through an HTTP API.

This repository takes a data-preprocessing + model-training pipeline and wraps
it in the operational components a production ML system needs:

- **[Dagster](https://dagster.io)** orchestrates the asset graph (raw CSVs →
  cleaned dataset → engineered features → trained model).
- **[MLflow](https://mlflow.org)** tracks every training run (parameters,
  metrics, artifacts) and hosts the model registry.
- **[LakeFS](https://lakefs.io)** versions the datasets and derived assets with
  a Git-like branching workflow, so you can always answer *"which data version
  produced this model?"*
- **[FastAPI](https://fastapi.tiangolo.com)** serves the model that currently
  holds the `production` alias in the registry — never a hardcoded file.

---

## Architecture at a glance

```
 raw CSVs ──► [ Dagster pipeline ]──────────────────────────────►  trained_model
 (LakeFS       hourly_rentals → time_features → weather →            (XGBoost +
  main          final_dataset → engineered_features → trained_model    log1p)
  branch)                                          │
                                                   ├─► MLflow run  (params, metrics,
                                                   │   + registry    artifact, alias=candidate)
                                                   └─► LakeFS commit (output branch:
                                                       processed CSVs + model pickle)

   promote_model.py   ── moves the `candidate` alias ──►  `production`

   FastAPI /predict   ── loads models:/<name>@production from MLflow ──►  demand forecast
```

The pipeline can read/write three storage backends, selected by the
`STORAGE_BACKEND` environment variable:

| Backend  | Raw inputs read from        | Outputs written to                         | Versioning |
| -------- | --------------------------- | ------------------------------------------ | ---------- |
| `local`  | `data/raw/`                 | `data/output-local/`                       | none       |
| `s3`     | `data/raw/`                 | `s3://<bucket>/processed` (RustFS)          | none       |
| `lakefs` | `lakefs://<repo>/main/raw`  | `lakefs://<repo>/<output>/processed` + commit | **LakeFS** |

`local` is the zero-dependency default; `lakefs` is the full MLOps setup and is
what the committed `.env` is configured for.

---

## Quick start

The fastest way to see a prediction come out the other end. The full path uses
the LakeFS backend, which needs Docker. If you just want the pipeline without
data versioning, see [Running with the `local` backend](#running-with-the-local-backend).

### Prerequisites

- **[uv](https://docs.astral.sh/uv/)** — Python package/runtime manager.
- **Docker** — runs the RustFS (object store) and LakeFS containers.
- Python 3.10–3.14 (uv installs an interpreter for you if needed).

### 1. Install dependencies

```bash
uv sync
```

### 2. Start the infrastructure

These are long-running services — run each in its own terminal (or background
them; the `start-*` scripts log to `logs/`).

```bash
./start-rustfs.sh     # S3-compatible object store  → :9000 (API), :9001 (UI)
./start-lakefs.sh     # data versioning server       → :8000 (UI)
./start-mlflow.sh     # tracking server + registry   → :5000 (UI)
```

### 3. Bootstrap LakeFS (once)

Creates the RustFS bucket, the LakeFS repo, uploads the raw CSVs to `main`,
creates the `output` branch, and protects `main` from direct writes:

```bash
uv run python scripts/lakefs_init.py
```

### 4. Run the pipeline and train a model

```bash
uv run dg dev        # Dagster UI → http://localhost:3000
```

In the Dagster UI, materialize all assets. This cleans the data, engineers
features, trains the XGBoost model, logs the run to MLflow (registering a new
model version with the `candidate` alias), and commits the derived assets to the
LakeFS `output` branch.

### 5. Promote the model to production

Review the run's metrics in the MLflow UI (`:5000`), then promote the candidate:

```bash
uv run python scripts/promote_model.py            # promote current `candidate`
# uv run python scripts/promote_model.py --version 7   # or a specific version
```

### 6. Serve API

```bash
./start-api-svc.sh    # FastAPI → http://localhost:8800 (loads the production model)
```

Interactive API docs are at <http://localhost:8800/docs>.

---

## Using the API

The API loads `models:/<registered-name>@production` from MLflow at startup and
exposes three endpoints:

| Method & path | Purpose |
| ------------- | ------- |
| `GET /health`  | Liveness + whether a model is loaded |
| `GET /model`   | Which registry version/run is currently served |
| `POST /predict`| Predict demand for one hour |

The request schema is intentionally **business-friendly**: the caller supplies a
timestamp, weather, a holiday flag, and the two historical rental counts the
model relies on. The service derives the cyclic/interaction features
(`hour_sin`, `is_peak_hour`, …) internally, exactly as the training pipeline
does — so callers don't need to know the model's feature schema.

```bash
curl -X POST http://localhost:8800/predict \
  -H 'Content-Type: application/json' \
  -d '{
        "timestamp": "2024-06-15T08:00:00",
        "conditions": "clear",
        "is_holiday": false,
        "lag_1h": 120,
        "lag_168h": 340
      }'
```

```jsonc
{
  "predicted_total_count": 287,
  "model_name": "xgboost-log1p-bike-rental-demand",
  "model_version": "3",
  "model_alias": "production"
}
```

| Field        | Type                                              | Notes |
| ------------ | ------------------------------------------------- | ----- |
| `timestamp`  | ISO 8601 datetime                                 | Only hour + day-of-week are used |
| `conditions` | `clear` \| `clouds` \| `light_rain` \| `heavy_rain` | `*_rain` collapse to `rain` (matching training) |
| `is_holiday` | bool (default `false`)                            | Public-holiday flag |
| `lag_1h`     | float ≥ 0                                         | Observed rentals 1 hour earlier |
| `lag_168h`   | float ≥ 0                                         | Observed rentals 168 hours (7 days) earlier |

---

## The model

`trained_model` fits a `TransformedTargetRegressor` wrapping an XGBoost
regressor, with a `log1p` / `expm1` transform on the target (rental counts are
right-skewed). A **chronological 80/20 split** is used so the holdout mirrors the
deployment setting — predicting future hours from past ones.

Reference metrics on the test split: **RMSE ≈ 45.5, MAE ≈ 28.6, R² ≈ 0.96**.

Features: `hour`, `day_of_week`, `is_weekend`, `is_holiday`, the cyclic
`hour_sin`/`hour_cos`, an `is_peak_hour` interaction flag, the `lag_1h` /
`lag_168h` autocorrelation lags, and one-hot encoded `conditions`. The modelling
rationale lives in the [`notebooks/`](notebooks/) (EDA → baseline → improvement).

---

## Design decisions

### Experiment organization (MLflow)
All training runs land in a single experiment (`bike-rental-demand`). Each run is
named `dagster-<run-id>` and tagged with its Dagster run id, model family, target
transform, and — when LakeFS is active — the **source-branch commit SHA of the
data it trained on**. That tag is the link that makes "which data produced this
model?" answerable directly from the MLflow UI.

### Model versioning (MLflow registry)
Every trained model is registered under one name
(`xgboost-log1p-bike-rental-demand`) and immediately tagged with the
**`candidate`** alias. Promotion to **`production`** is a deliberate, separate
step (`scripts/promote_model.py`) taken after a human reviews the metrics. The
API only ever loads `@production`, so this one alias move is the single source of
truth for what users are served — no redeploy required.

### Data & asset versioning (LakeFS)
A deliberately simple two-branch strategy:

- **`main`** — protected source branch holding the raw inputs. A
  branch-protection rule (set by `lakefs_init.py`) blocks direct writes; the only
  way data enters `main` is a reviewed **merge** from `output`.
- **`output`** — where each pipeline run writes its derived CSVs and the model
  pickle. The `versioned_outputs` asset commits those at the end of every run, so
  derived assets are versioned alongside the model that produced them.

This answers both questions the brief poses: *which dataset version trained this
model?* (the commit SHA tagged on the MLflow run) and *how is `main` protected?*
(branch protection + merge-only updates).

---

## Configuration

Configuration is read from the `.env` file at the repo root (loaded by the app,
the scripts, and the `start-*.sh` scripts). The committed `.env` is a working
local-development setup (`admin`/`admin` credentials, all services on
`localhost`). Key variables:

| Variable | Default | Used by |
| -------- | ------- | ------- |
| `STORAGE_BACKEND` | `local` | Selects `local` / `s3` / `lakefs` |
| `RAW_DATA_DIR` | `data/raw` | Pipeline, LakeFS init |
| `LOCAL_STORAGE_DIR` | `data/output-local` | `local` backend outputs |
| `MLFLOW_TRACKING_URI` | `http://localhost:5000` | Pipeline, API, promote script |
| `MLFLOW_EXPERIMENT_NAME` | `bike-rental-demand` | Training runs grouped under this |
| `MLFLOW_REGISTERED_MODEL_NAME` | `xgboost-log1p-bike-rental-demand` | Registry name the API loads |
| `LAKEFS_REPO` | `repo` | LakeFS repository name |
| `LAKEFS_SOURCE_BRANCH` | `main` | Protected read branch |
| `LAKEFS_OUTPUT_BRANCH` | `output` | Write/commit branch |
| `LAKEFS_ENDPOINT_URL` | `http://localhost:8000` | LakeFS server |
| `RUSTFS_BUCKET` | `bucket` | Object-store bucket (backs LakeFS + `s3` mode) |
| `RUSTFS_ENDPOINT_URL` | `http://localhost:9000` | RustFS S3 endpoint |

(Defaults shown are the code-level fallbacks; the committed `.env` may set
different values, e.g. `STORAGE_BACKEND=lakefs`.)

### Running with the `local` backend

No Docker, no LakeFS — fastest for iterating on pipeline logic. You still need
MLflow running for training and serving:

```bash
# set STORAGE_BACKEND=local in .env (or `export STORAGE_BACKEND=local`)
./start-mlflow.sh
uv run dg dev               # materialize assets → data/output-local/
uv run python scripts/promote_model.py
./start-api-svc.sh
```

The `versioned_outputs` asset becomes a no-op in this mode (nothing to commit),
so the asset graph stays identical across backends.

---

## Project structure

```
.
├── api/
│   └── main.py                     # FastAPI service (loads @production model)
├── scripts/
│   ├── lakefs_init.py              # one-time LakeFS repo/branch/protection bootstrap
│   └── promote_model.py            # move `candidate` → `production` alias
├── src/bike_rental/
│   ├── definitions.py              # Dagster Definitions: wires assets + resources
│   └── defs/
│       ├── assets/                 # one module per asset in the graph
│       │   ├── hourly.py           #  → hourly_rentals
│       │   ├── time_features.py    #  → rentals_with_time_features
│       │   ├── weather.py          #  → rentals_with_weather
│       │   ├── final_dataset.py    #  → final_dataset
│       │   ├── features.py         #  → engineered_features
│       │   ├── model.py            #  → trained_model (+ MLflow logging/registry)
│       │   └── versioned_outputs.py#  → commits derived assets to LakeFS
│       ├── io_managers/            # CSV + pickle IO managers (local / S3 / LakeFS)
│       └── resources/              # DataConfig, LakeFSResource, MLflowResource
├── notebooks/                      # EDA + modelling exploration (00–03)
├── data/raw/                       # source CSVs (rentals, weather, holidays)
├── start-rustfs.sh / start-lakefs.sh / start-mlflow.sh / start-api-svc.sh
├── pyproject.toml                  # deps + uv/dg/ruff config
└── .env                            # configuration
```

---

## Data

Four raw CSVs under `data/raw/` cover **Jan 2011 – Dec 2012** (17,379 hours):

- `registered_bike_rentals.csv` / `direct_pickup_bike_rentals.csv` — individual
  trip records (one row per trip), aggregated to hourly totals.
- `weather.csv` — hourly conditions, temperature, humidity, wind.
- `holidays.csv` — public-holiday calendar.

---

## Tooling

```bash
uv run ruff check .       # lint (rules: E, W, F, I, UP, D — numpy docstrings)
uv run ruff format .      # format
```

"""FastAPI service for bike rental demand prediction.

Loads the model carrying the ``production`` alias from the MLflow model
registry at startup and serves predictions through ``POST /predict``.

The endpoint accepts a small business-friendly schema — timestamp,
weather conditions, holiday flag, and the two lag observations the
estimator needs (rentals 1h ago and 168h ago). The service derives the
cyclic / interaction features (``hour_sin``, ``hour_cos``,
``is_peak_hour``, ...) the same way the training pipeline does so the
caller doesn't have to know about them.

The model is never read from a local file: only ``models:/<name>@production``
URIs are used. Promoting a new version into production is therefore the
single source of truth for what this API serves (see
``scripts/promote_model.py``).

The served alias is fixed to ``production`` (the ``MODEL_ALIAS`` constant):
this service exists to serve the production model, and promotion is the
only control over what it loads.

Environment variables
---------------------
``MLFLOW_TRACKING_URI``           — MLflow tracking server URI.
``MLFLOW_REGISTERED_MODEL_NAME``  — name of the registered model.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime
from enum import StrEnum
from os import getenv
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MLFLOW_REGISTERED_MODEL_NAME = getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "xgboost-log1p-bike-rental-demand"
)
_MLFLOW_TRACKING_URI = getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_ALIAS = "production"

_state: dict[str, Any] = {}


def _load_model() -> None:
    """Resolve the production alias and load the corresponding model."""
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(
        _MLFLOW_REGISTERED_MODEL_NAME, MODEL_ALIAS
    )
    model_uri = f"models:/{_MLFLOW_REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    _state["model"] = mlflow.sklearn.load_model(model_uri)
    _state["model_uri"] = model_uri
    _state["model_version"] = mv.version
    _state["model_run_id"] = mv.run_id


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the production model on startup; release it on shutdown."""
    _load_model()
    yield
    _state.clear()


app = FastAPI(
    title="Bike rental demand predictor",
    description=(
        "Predicts hourly bike rental demand. Loads the model carrying "
        "the 'production' alias from the MLflow registry at startup."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class WeatherCondition(StrEnum):
    """Weather conditions accepted by the prediction endpoint."""

    CLEAR = "clear"
    CLOUDS = "clouds"
    LIGHT_RAIN = "light_rain"
    HEAVY_RAIN = "heavy_rain"


class PredictRequest(BaseModel):
    """Single-hour prediction request.

    The lag fields are required because the model has weekly + hourly
    autocorrelation baked in — they should come from the operator's
    historical rental counts.
    """

    timestamp: datetime = Field(
        ...,
        description=(
            "Hour for which to predict demand (ISO 8601). Only the hour "
            "and the day-of-week are used."
        ),
        examples=["2024-06-15T08:00:00"],
    )
    conditions: WeatherCondition = Field(
        ...,
        description=(
            "Weather conditions. One of 'clear', 'clouds', "
            "'light_rain' or 'heavy_rain'."
        ),
        examples=["clear", "clouds", "light_rain", "heavy_rain"],
    )
    is_holiday: bool = Field(
        default=False,
        description="Whether the day is a public holiday.",
    )
    lag_1h: float = Field(
        ..., ge=0, description="Observed total rentals 1 hour earlier."
    )
    lag_168h: float = Field(
        ...,
        ge=0,
        description="Observed total rentals 168 hours (7 days) earlier.",
    )


class PredictResponse(BaseModel):
    """Predicted demand plus the registry coordinates of the served model."""

    predicted_total_count: int = Field(
        ..., description="Predicted hourly rentals, rounded and clamped to ≥0."
    )
    model_name: str
    model_version: str
    model_alias: str


def _normalize_condition(condition: WeatherCondition) -> str:
    if condition in {
        WeatherCondition.LIGHT_RAIN,
        WeatherCondition.HEAVY_RAIN,
    }:
        return "rain"

    return condition.value


def _features_for(req: PredictRequest) -> pd.DataFrame:
    """Mirror the engineering in ``defs/assets/features.py`` for one row.

    Kept in sync with the training pipeline's feature transformations so
    the API speaks the same feature schema as the trained estimator.
    """
    ts = req.timestamp
    hour = ts.hour
    day_of_week = ts.weekday()
    is_weekend = int(day_of_week >= 5)
    is_holiday = int(req.is_holiday)
    hour_sin = float(np.sin(2 * np.pi * hour / 24))
    hour_cos = float(np.cos(2 * np.pi * hour / 24))
    is_peak_hour = int(
        (hour in {7, 8, 9, 17, 18, 19} and not is_weekend)
        or (12 <= hour <= 16 and is_weekend)
    )
    return pd.DataFrame(
        [
            {
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "is_holiday": is_holiday,
                "hour_sin": hour_sin,
                "hour_cos": hour_cos,
                "is_peak_hour": is_peak_hour,
                "lag_1h": req.lag_1h,
                "lag_168h": req.lag_168h,
                "conditions": _normalize_condition(req.conditions),
            }
        ]
    )


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness probe reporting whether a model is currently loaded."""
    return {"status": "ok", "model_loaded": "model" in _state}


@app.get("/model")
def model_info() -> dict[str, Any]:
    """Return the registry coordinates of the currently served model."""
    if "model" not in _state:
        raise HTTPException(503, "Model not loaded")
    return {
        "name": _MLFLOW_REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS,
        "version": _state["model_version"],
        "run_id": _state["model_run_id"],
        "uri": _state["model_uri"],
        "tracking_uri": _MLFLOW_TRACKING_URI,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Predict demand for one hour, clamped to a non-negative integer."""
    model = _state.get("model")
    if model is None:
        raise HTTPException(503, "Model not loaded")
    features = _features_for(req)
    raw = float(model.predict(features)[0])
    return PredictResponse(
        predicted_total_count=max(0, round(raw)),
        model_name=_MLFLOW_REGISTERED_MODEL_NAME,
        model_version=str(_state["model_version"]),
        model_alias=MODEL_ALIAS,
    )

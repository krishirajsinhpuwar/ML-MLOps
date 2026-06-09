"""Dagster Definitions — wires assets, resources, and IO manager together.

The destination for both raw inputs and processed outputs is selected
via the ``STORAGE_BACKEND`` environment variable:

- ``local`` (default) — raw inputs read from ``<repo>/data/raw``; outputs
  written to ``<repo>/data/output-local``.
- ``s3`` — raw inputs read from ``<repo>/data/raw``; outputs written to
  an S3-compatible bucket (e.g. RustFS). Endpoint, credentials, and
  bucket name come from ``RUSTFS_*`` environment variables.
- ``lakefs`` — raw inputs read from a LakeFS repository at
  ``lakefs://<repo>/<source_branch>/raw``; outputs written to
  ``lakefs://<repo>/<output_branch>/processed`` and committed by the
  ``versioned_outputs`` asset. Endpoint, credentials, repo name, and
  branch names come from ``LAKEFS_*`` environment variables.
"""

from os import getenv
from pathlib import Path

from dagster import Definitions, definitions
from dotenv import load_dotenv

from bike_rental.defs.assets import (
    engineered_features,
    final_dataset,
    hourly_rentals,
    rentals_with_time_features,
    rentals_with_weather,
    trained_model,
    versioned_outputs,
)
from bike_rental.defs.io_managers.csv_io_manager import CSVIOManager
from bike_rental.defs.io_managers.pickle_io_manager import PickleIOManager
from bike_rental.defs.jobs import retrain_job
from bike_rental.defs.resources.config import DataConfig
from bike_rental.defs.resources.lakefs import LakeFSResource
from bike_rental.defs.resources.mlflow import MLflowResource
from bike_rental.defs.sensors import lakefs_source_data_sensor

# Resolve paths relative to this file so the pipeline works from any cwd
_PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment variables from .env file (if it exists)
load_dotenv(_PROJECT_ROOT / ".env")


_BACKEND = getenv("STORAGE_BACKEND", "local").lower()
_RAW_DATA_DIR = getenv("RAW_DATA_DIR", "data/raw")
_LOCAL_STORAGE_DIR = getenv("LOCAL_STORAGE_DIR", "data/output-local")

_RUSTFS_BUCKET = getenv("RUSTFS_BUCKET", "bucket")
_RUSTFS_ENDPOINT_URL = getenv("RUSTFS_ENDPOINT_URL", "http://localhost:9000")
_RUSTFS_ACCESS_KEY = getenv("RUSTFS_ACCESS_KEY", "admin")
_RUSTFS_SECRET_KEY = getenv("RUSTFS_SECRET_KEY", "admin")

_LAKEFS_REPO = getenv("LAKEFS_REPO", "repo")
_LAKEFS_SOURCE_BRANCH = getenv("LAKEFS_SOURCE_BRANCH", "main")
_LAKEFS_OUTPUT_BRANCH = getenv("LAKEFS_OUTPUT_BRANCH", "output")
_LAKEFS_ENDPOINT_URL = getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000")
_LAKEFS_ACCESS_KEY = getenv("LAKEFS_ACCESS_KEY", "admin")
_LAKEFS_SECRET_KEY = getenv("LAKEFS_SECRET_KEY", "admin")

_MLFLOW_TRACKING_URI = getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
_MLFLOW_EXPERIMENT_NAME = getenv("MLFLOW_EXPERIMENT_NAME", "bike-rental-demand")


def _build_config() -> tuple[str, str, dict | None]:
    """Resolve ``(data_dir, output_dir, storage_options)`` from env vars.

    The triple is consumed by the ``DataConfig`` resource and the IO
    managers. For the LakeFS backend, raw inputs and outputs live on
    different branches of the same repo (``source_branch`` → read,
    ``output_branch`` → write).
    """
    if _BACKEND == "s3":
        data_dir = str(_PROJECT_ROOT / _RAW_DATA_DIR)
        output_dir = f"s3://{_RUSTFS_BUCKET}/processed"
        storage_options = {
            "key": _RUSTFS_ACCESS_KEY,
            "secret": _RUSTFS_SECRET_KEY,
            "client_kwargs": {
                "endpoint_url": _RUSTFS_ENDPOINT_URL,
            },
        }
    elif _BACKEND == "lakefs":
        data_dir = f"lakefs://{_LAKEFS_REPO}/{_LAKEFS_SOURCE_BRANCH}/raw"
        output_dir = (
            f"lakefs://{_LAKEFS_REPO}/{_LAKEFS_OUTPUT_BRANCH}/processed"
        )
        storage_options = {
            "host": _LAKEFS_ENDPOINT_URL,
            "username": _LAKEFS_ACCESS_KEY,
            "password": _LAKEFS_SECRET_KEY,
        }
    else:
        data_dir = str(_PROJECT_ROOT / _RAW_DATA_DIR)
        output_dir = str(_PROJECT_ROOT / _LOCAL_STORAGE_DIR)
        storage_options = None

    return data_dir, output_dir, storage_options


_DATA_DIR, _OUTPUT_DIR, _STORAGE_OPTIONS = _build_config()


@definitions
def defs() -> Definitions:
    """Construct the Dagster Definitions object.

    The assets are defined in separate modules under ``defs/assets/`` and
    imported here. The resources include a ``DataConfig`` that encapsulates
    paths and storage options, and a custom ``CSVIOManager`` that handles
    reading/writing CSVs to the configured backend.

    ``retrain_job`` materializes the whole graph; ``lakefs_source_data_sensor``
    (the bonus) launches that job automatically when the LakeFS source data
    changes — see ``defs/sensors.py``.

    Returns
    -------
        ``dagster.Definitions`` object with all assets, jobs, sensors, and
        resources wired together. This is the entry point for Dagster to
        discover the pipeline components.

    """
    return Definitions(
        assets=[
            hourly_rentals,
            rentals_with_time_features,
            rentals_with_weather,
            final_dataset,
            engineered_features,
            trained_model,
            versioned_outputs,
        ],
        jobs=[retrain_job],
        sensors=[lakefs_source_data_sensor],
        resources={
            "data_config": DataConfig(
                data_dir=_DATA_DIR,
                output_dir=_OUTPUT_DIR,
                storage_options=_STORAGE_OPTIONS,
            ),
            "io_manager": CSVIOManager(
                output_dir=_OUTPUT_DIR,
                storage_options=_STORAGE_OPTIONS,
            ),
            "pickle_io_manager": PickleIOManager(
                output_dir=_OUTPUT_DIR,
                storage_options=_STORAGE_OPTIONS,
            ),
            "mlflow_config": MLflowResource(
                tracking_uri=_MLFLOW_TRACKING_URI,
                experiment_name=_MLFLOW_EXPERIMENT_NAME,
            ),
            "lakefs": LakeFSResource(
                enabled=_BACKEND == "lakefs",
                storage_options=_STORAGE_OPTIONS,
                repo=_LAKEFS_REPO,
                source_branch=_LAKEFS_SOURCE_BRANCH,
                output_branch=_LAKEFS_OUTPUT_BRANCH,
            ),
        },
    )

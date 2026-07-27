"""Dagster Definitions — wires assets, resources, and IO manager together.

Raw CSV inputs are always read from ``<repo>/data/raw``. The destination
for processed outputs is selected via the ``STORAGE_BACKEND`` environment
variable:

- ``local`` (default) — outputs are written to ``<repo>/data/output-local``.
- ``s3`` — outputs are written to an S3-compatible bucket (e.g. RustFS).
  The endpoint, credentials, and bucket name are controlled by
  ``RUSTFS_*`` environment variables.
"""

from os import getenv
from pathlib import Path

from dagster import Definitions
from dotenv import load_dotenv

from bike_rental.defs.assets import (
    final_dataset,
    hourly_rentals,
    rentals_with_time_features,
    rentals_with_weather,
)
from bike_rental.defs.io_managers.csv_io_manager import CSVIOManager
from bike_rental.defs.resources.config import DataConfig

# Resolve paths relative to this file so the pipeline works from any cwd
_PROJECT_ROOT = Path(__file__).parent.parent

# Load environment variables from .env file (if it exists)
load_dotenv(_PROJECT_ROOT / ".env")

_DATA_DIR = str(_PROJECT_ROOT / getenv("RAW_DATA_DIR", "data/raw"))


def _build_config() -> tuple[str, dict | None]:
    """Resolve the (output_dir, storage_options) pair based on environment variables.

    Returns
    -------
        ``(output_dir, storage_options)``. ``storage_options``
        is ``None`` for the local backend and a pandas-compatible mapping
        for the S3 backend.
    """
    backend = getenv("STORAGE_BACKEND", "local").lower()
    if backend == "s3":
        output_dir = f"s3://{getenv('RUSTFS_BUCKET', 'assets')}"
        storage_options = {
            "key": getenv("RUSTFS_ACCESS_KEY", "rustfsadmin"),
            "secret": getenv("RUSTFS_SECRET_KEY", "rustfsadmin"),
            "client_kwargs": {
                "endpoint_url": getenv(
                    "RUSTFS_ENDPOINT_URL", "http://localhost:9000"
                ),
            },
        }
    else:
        output_dir = str(
            _PROJECT_ROOT / getenv("LOCAL_STORAGE_DIR", "data/output-local")
        )
        storage_options = None

    return output_dir, storage_options


_OUTPUT_DIR, _STORAGE_OPTIONS = _build_config()


defs = Definitions(
    assets=[
        hourly_rentals,
        rentals_with_time_features,
        rentals_with_weather,
        final_dataset,
    ],
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
    },
)

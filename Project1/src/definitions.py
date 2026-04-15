"""Dagster Definitions — wires assets, resources, and IO manager together."""

from pathlib import Path
from dagster import Definitions

from src.assets import (
    final_dataset,
    hourly_rentals,
    rentals_with_time_features,
    rentals_with_weather,
)
from src.io_managers.csv_io_manager import CSVIOManager
from src.resources.config import DataConfig


# Resolve paths relative to this file so the pipeline works from any cwd
_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_DIR = str(_PROJECT_ROOT / "data")
_OUTPUT_DIR = str(_PROJECT_ROOT / "output")

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
        ),
        "io_manager": CSVIOManager(
            output_dir=_OUTPUT_DIR
        ),
    },
)

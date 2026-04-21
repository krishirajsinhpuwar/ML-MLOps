"""Dagster Definitions — wires assets, resources, and IO manager together."""

from pathlib import Path
from dagster import Definitions

from pipeline.assets import (
    hourly_rentals,
    rentals_with_time_features,
    rentals_with_weather,
    final_dataset,
)
from pipeline.io_managers.csv_io_manager import CSVIOManager
from pipeline.resources.config import DataConfig


# Resolve paths relative to this file so the pipeline works from any cwd
_PROJECT_ROOT = Path(__file__).parent.parent
_DATA_DIR = str(_PROJECT_ROOT / "raw_data")
_OUTPUT_DIR = str(_PROJECT_ROOT / "data")

defs = Definitions(
    assets=[
        hourly_rentals,
        rentals_with_time_features,
        rentals_with_weather,
        final_dataset,
    ],
    resources={
        "data_config": DataConfig(data_dir=_DATA_DIR, output_dir=_OUTPUT_DIR),
        "io_manager": CSVIOManager(output_dir=_OUTPUT_DIR),
    },
)

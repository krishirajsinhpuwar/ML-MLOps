"""Dagster assets for the bike-sharing preprocessing pipeline."""

from pipeline.assets.final_dataset import final_dataset
from pipeline.assets.hourly import hourly_rentals
from pipeline.assets.time_features import rentals_with_time_features
from pipeline.assets.weather import rentals_with_weather

__all__ = [
    "hourly_rentals",
    "rentals_with_time_features",
    "rentals_with_weather",
    "final_dataset",
]

"""Dagster assets for the bike-sharing preprocessing pipeline."""

from src.assets.final_dataset import final_dataset
from src.assets.rentals import hourly_rentals
from src.assets.time_features import rentals_with_time_features
from src.assets.weather import rentals_with_weather

__all__ = [
    "hourly_rentals",
    "rentals_with_time_features",
    "rentals_with_weather",
    "final_dataset",
]

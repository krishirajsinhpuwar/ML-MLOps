"""Dagster assets for the bike-sharing preprocessing pipeline."""

from bike_rental.defs.assets.features import engineered_features
from bike_rental.defs.assets.final_dataset import final_dataset
from bike_rental.defs.assets.hourly import hourly_rentals
from bike_rental.defs.assets.model import trained_model
from bike_rental.defs.assets.time_features import rentals_with_time_features
from bike_rental.defs.assets.versioned_outputs import versioned_outputs
from bike_rental.defs.assets.weather import rentals_with_weather

__all__ = [
    "hourly_rentals",
    "rentals_with_time_features",
    "rentals_with_weather",
    "final_dataset",
    "engineered_features",
    "trained_model",
    "versioned_outputs",
]

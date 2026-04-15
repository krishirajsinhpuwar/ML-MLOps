"""Asset: derive time-based features from the hourly rental timestamp."""

import pandas as pd
from dagster import asset


@asset
def rentals_with_time_features(hourly_rentals: pd.DataFrame) -> pd.DataFrame:
    """Engineer temporal features from the ``hour`` timestamp column.

    Adds columns that expose cyclical and categorical time signals useful for
    machine learning: hour of day, day of week, month, year, and a weekend
    flag.

    Parameters
    ----------
    hourly_rentals : pd.DataFrame
        Output of the ``hourly_rentals`` asset. Must contain an ``hour``
        column of dtype ``datetime64``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame extended with columns:
        ``hour_of_day``, ``day_of_week``, ``day``, ``month``, ``year``, ``is_weekend``.
    """
    df = hourly_rentals.copy()

    df["hour_of_day"] = df["hour"].dt.hour  # 0–23
    df["day_of_week"] = df["hour"].dt.dayofweek  # 0=Monday, 6=Sunday
    df["day"] = df["hour"].dt.day  # 1–31
    df["month"] = df["hour"].dt.month  # 1–12
    df["year"] = df["hour"].dt.year
    df["is_weekend"] = df["day_of_week"] >= 5  # Saturday or Sunday

    return df

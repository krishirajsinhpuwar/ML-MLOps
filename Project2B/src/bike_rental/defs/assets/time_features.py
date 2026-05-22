"""Asset: derive time-based features from the hourly rental timestamp."""

import pandas as pd
from dagster import asset


@asset
def rentals_with_time_features(
    context, hourly_rentals: pd.DataFrame
) -> pd.DataFrame:
    """Engineer temporal features from the ``datetime`` column.

    Adds columns that expose calendar and time-of-day signals useful for
    machine learning: hour of day, day-of-month, month, year, ISO week,
    day-of-week, a weekend flag, and the calendar date used as the join
    key for the holiday calendar.

    Parameters
    ----------
    hourly_rentals : pd.DataFrame
        Output of the ``hourly_rentals`` asset. Must contain a ``datetime``
        column of dtype ``datetime64``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame extended with columns:
        ``hour``, ``day_of_month``, ``month``, ``year``, ``week``,
        ``day_of_week``, ``is_weekend``, ``date``.

    """
    df = hourly_rentals.copy()

    df["hour"] = df["datetime"].dt.hour.astype("int8")  # 0–23
    df["day_of_month"] = df["datetime"].dt.day.astype("int8")  # 1–31
    df["month"] = df["datetime"].dt.month.astype("int8")  # 1–12
    df["year"] = df["datetime"].dt.year.astype("int16")  # e.g. 2011
    df["week"] = df["datetime"].dt.isocalendar().week.astype("int8")  # 1–53
    df["day_of_week"] = df["datetime"].dt.dayofweek.astype(
        "int8"
    )  # 0=Monday, 6=Sunday
    df["is_weekend"] = df["day_of_week"] >= 5  # Saturday or Sunday

    df["date"] = df["datetime"].dt.date.astype(
        "datetime64[us]"
    )  # calendar date, for holiday join

    context.log.info(
        f"Time features engineered. Sample values:\n"
        f"{df.head()}\nInfo:\n{df.info()}"
    )

    return df

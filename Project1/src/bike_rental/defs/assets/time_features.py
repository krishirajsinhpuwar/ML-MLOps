"""Asset: derive time-based features from the hourly rental timestamp."""

import pandas as pd
from dagster import asset


@asset
def rentals_with_time_features(
    context, hourly_rentals: pd.DataFrame
) -> pd.DataFrame:
    """Engineer temporal features from the ``datetime`` column.

    Adds columns that expose categorical time signals useful for machine
    learning: hour of day, date, day of week, a weekend flag, and a coarse
    time-of-day bucket.

    Parameters
    ----------
    hourly_rentals : pd.DataFrame
        Output of the ``hourly_rentals`` asset. Must contain a ``datetime``
        column of dtype ``datetime64``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame extended with columns:
        ``hour``, ``date``, ``day``, ``is_weekend``, ``time_of_day``.
    """
    df = hourly_rentals.copy()

    df["hour"] = df["datetime"].dt.hour.astype("int8")  # 0–23
    df["date"] = df["datetime"].dt.date.astype(
        "datetime64[us]"
    )  # calendar date, for holiday join
    df["day"] = df["datetime"].dt.dayofweek.astype("int8")  # 0=Monday, 6=Sunday
    df["is_weekend"] = df["day"] >= 5  # Saturday or Sunday
    df["time_of_day"] = pd.cut(
        df["hour"],
        # 0=night[0-5], 1=morning[6-11], 2=afternoon[12-17], 3=evening[18-23]
        bins=[-1, 5, 11, 17, 23],
        labels=[0, 1, 2, 3],
    ).astype("int8")

    context.log.info(
        f"Time features engineered. Sample values:\n"
        f"{df.head()}\nInfo:\n{df.info()}"
    )

    return df

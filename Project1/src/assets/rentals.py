"""Asset: load and aggregate booked and direct rental records to hourly totals."""

import pandas as pd
from dagster import asset

from src.resources.config import DataConfig


@asset(required_resource_keys={"data_config"})
def hourly_rentals(context) -> pd.DataFrame:
    """Load registered and direct rental CSVs, aggregate both to hourly counts.

    Individual rental records (one row per trip) are floored to the hour and
    counted. The two sources are outer-merged so that no hour is silently
    dropped if it appears in only one source.

    Parameters
    ----------
    context : dagster.OpExecutionContext
        Dagster execution context providing access to resources.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ``hour``, ``registered_count``, ``direct_count``, ``total_count``.
    """
    cfg: DataConfig = context.resources.data_config

    registered = pd.read_csv(
        cfg.get_data_path("registered_bike_rentals.csv"),
        parse_dates=["datetime"],
    )
    direct = pd.read_csv(
        cfg.get_data_path("direct_pickup_bike_rentals.csv"),
        parse_dates=["datetime"],
    )

    registered["hour"] = registered["datetime"].dt.floor("h")
    direct["hour"] = direct["datetime"].dt.floor("h")

    registered_hourly = registered.groupby("hour").size().reset_index(name="registered_count")
    direct_hourly = direct.groupby("hour").size().reset_index(name="direct_count")

    df = pd.merge(registered_hourly, direct_hourly, on="hour", how="outer").fillna(0)
    df["registered_count"] = df["registered_count"].astype(int)
    df["direct_count"] = df["direct_count"].astype(int)
    df["total_count"] = df["registered_count"] + df["direct_count"]
    df = df.sort_values("hour").reset_index(drop=True)

    context.log.info(
        f"Hourly rentals: {len(df)} rows, range {df['hour'].min()} → {df['hour'].max()}"
    )
    return df

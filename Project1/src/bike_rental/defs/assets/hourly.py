"""Asset: load and aggregate booked and direct rental records to hourly totals."""

import pandas as pd
from dagster import asset

from bike_rental.defs.resources.config import DataConfig


@asset(required_resource_keys={"data_config"})
def hourly_rentals(context) -> pd.DataFrame:
    """Load registered and direct rental CSVs, aggregate both to hourly counts.

    Individual rental records (one row per trip) are floored to the hour and
    counted. The two sources are outer-merged so that no hour is silently
    dropped if it appears in only one source. Input locations are resolved
    by ``data_config`` and may live on the local filesystem or in an
    S3-compatible bucket.

    Parameters
    ----------
    context : dagster.OpExecutionContext
        Dagster execution context providing access to resources.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        ``datetime``, ``registered_count``, ``direct_count``, ``total_count``.
    """
    cfg: DataConfig = context.resources.data_config

    registered_path = cfg.get_data_path("registered_bike_rentals.csv")
    direct_path = cfg.get_data_path("direct_pickup_bike_rentals.csv")

    registered = pd.read_csv(
        registered_path,
        parse_dates=["datetime"],
        index_col="id",
        storage_options=cfg.storage_options_for(registered_path),
    )
    direct = pd.read_csv(
        direct_path,
        parse_dates=["datetime"],
        index_col="id",
        storage_options=cfg.storage_options_for(direct_path),
    )

    registered["date_hour"] = registered["datetime"].dt.floor("h")
    direct["date_hour"] = direct["datetime"].dt.floor("h")

    registered_hourly = (
        registered.groupby("date_hour")
        .size()
        .reset_index(name="registered_count")
    )
    direct_hourly = (
        direct.groupby("date_hour").size().reset_index(name="direct_count")
    )

    registered_hourly.rename(columns={"date_hour": "datetime"}, inplace=True)
    direct_hourly.rename(columns={"date_hour": "datetime"}, inplace=True)

    df = pd.merge(
        registered_hourly, direct_hourly, on="datetime", how="outer"
    ).fillna(0)

    df["registered_count"] = df["registered_count"].astype(int)
    df["direct_count"] = df["direct_count"].astype(int)
    df["total_count"] = df["registered_count"] + df["direct_count"]

    df = df.sort_values("datetime").reset_index(drop=True)

    context.log.info(
        f"Hourly rentals loaded and aggregated. Sample values:\n{df.head()}\nInfo:\n{df.info()}"
    )

    return df

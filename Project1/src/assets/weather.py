"""Asset: enrich the hourly rental dataset with historical weather data."""

import pandas as pd
from dagster import asset

from src.resources.config import DataConfig


@asset(required_resource_keys={"data_config"})
def rentals_with_weather(context, rentals_with_time_features: pd.DataFrame) -> pd.DataFrame:
    """Left-join historical weather data onto the hourly rental dataset.

    Weather records are already aligned to exact hour boundaries, so no
    rounding is required. A left join is used to preserve all rental hours;
    any gaps in weather coverage will appear as NaN values.

    Parameters
    ----------
    context : dagster.OpExecutionContext
        Dagster execution context providing access to resources.
    rentals_with_time_features : pd.DataFrame
        Output of the ``rentals_with_time_features`` asset. Must contain an
        ``hour`` column of dtype ``datetime64``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with weather columns:
        ``conditions``, ``temperature_c``, ``perceived_temperature_c``,
        ``humidity``, ``windspeed_kmh``.
    """
    cfg: DataConfig = context.resources.data_config

    weather = pd.read_csv(
        cfg.get_data_path("weather.csv"),
        parse_dates=["datetime"],
    )
    weather = weather.drop(columns=["id"]).rename(columns={"datetime": "hour"})

    df = pd.merge(rentals_with_time_features, weather, on="hour", how="left")

    missing = df[["conditions", "temperature_c", "humidity", "windspeed_kmh"]].isna().sum().sum()
    context.log.info(f"Weather join complete. Missing weather values: {missing}. Shape: {df.shape}")
    return df

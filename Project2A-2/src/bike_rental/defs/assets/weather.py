"""Asset: enrich the hourly rental dataset with historical weather data."""

import pandas as pd
from dagster import asset

from bike_rental.defs.resources.config import DataConfig


@asset(required_resource_keys={"data_config"})
def rentals_with_weather(
    context, rentals_with_time_features: pd.DataFrame
) -> pd.DataFrame:
    """Left-join historical weather data onto the hourly rental dataset.

    Weather records are already aligned to exact hour boundaries, so no
    rounding is required. A left join is used to preserve all rental hours;
    any gaps in weather coverage will appear as NaN values.

    Parameters
    ----------
    context : dagster.OpExecutionContext
        Dagster execution context providing access to resources.
    rentals_with_time_features : pd.DataFrame
        Output of the ``rentals_with_time_features`` asset. Must contain a
        ``datetime`` column of dtype ``datetime64``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame enriched with weather columns:
        ``conditions``, ``temperature_c``, ``perceived_temperature_c``,
        ``humidity``, ``windspeed_kmh``.

    """
    cfg: DataConfig = context.resources.data_config

    weather_path = cfg.get_data_path("weather.csv")
    weather = pd.read_csv(
        weather_path,
        parse_dates=["datetime"],
        index_col="id",
        storage_options=cfg.storage_options_for(weather_path),
    )

    df = pd.merge(
        rentals_with_time_features, weather, on="datetime", how="left"
    )

    context.log.info(
        f"Weather data merged. Sample values:\n{df.head()}\nInfo:\n{df.info()}"
    )

    return df

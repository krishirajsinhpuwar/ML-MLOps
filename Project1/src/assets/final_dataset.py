"""Asset: merge holiday information and write the final prepared CSV."""

import pandas as pd
from dagster import asset

from src.resources.config import DataConfig


@asset(required_resource_keys={"data_config"})
def final_dataset(context, rentals_with_weather: pd.DataFrame) -> pd.DataFrame:
    """Add a holiday flag and produce the final ML-ready dataset.

    The holiday calendar is merged on the date portion of the ``hour``
    timestamp. A boolean ``is_holiday`` column is derived from the presence
    of a holiday name and the raw name column is then dropped.

    Parameters
    ----------
    context : dagster.OpExecutionContext
        Dagster execution context providing access to resources.
    rentals_with_weather : pd.DataFrame
        Output of the ``rentals_with_weather`` asset. Must contain an
        ``hour`` column of dtype ``datetime64``.

    Returns
    -------
    pd.DataFrame
        Fully prepared dataset with all rental, time-feature, weather, and
        holiday columns. Written to ``<output_dir>/final_dataset.csv`` via
        the IO manager.
    """
    cfg: DataConfig = context.resources.data_config

    holidays = pd.read_csv(
        cfg.get_data_path("holidays.csv"),
        parse_dates=["date"],
    )
    holidays = holidays.drop(columns=["id"])

    df = rentals_with_weather.copy()
    df["date"] = df["hour"].dt.normalize()

    df = pd.merge(df, holidays, on="date", how="left")
    df["is_holiday"] = df["holiday"].notna()
    df = df.drop(columns=["holiday", "date"])

    context.log.info(
        f"Final dataset ready. Shape: {df.shape}. "
        f"Holiday hours: {df['is_holiday'].sum()}. "
        f"Columns: {list(df.columns)}"
    )
    return df

"""Asset: merge holiday information and write the final prepared CSV."""

import pandas as pd
from dagster import asset

from pipeline.resources.config import DataConfig


@asset(required_resource_keys={"data_config"})
def final_dataset(context, rentals_with_weather: pd.DataFrame) -> pd.DataFrame:
    """Add a holiday flag and produce the final ML-ready dataset.

    The holiday calendar is merged on the ``date`` column produced by
    ``rentals_with_time_features``. A boolean ``is_holiday`` column is
    derived from the presence of a holiday name and the raw name column is
    then dropped.

    Parameters
    ----------
    context : dagster.OpExecutionContext
        Dagster execution context providing access to resources.
    rentals_with_weather : pd.DataFrame
        Output of the ``rentals_with_weather`` asset. Must contain a
        ``date`` column of dtype ``datetime64``.

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
        index_col="id",
    )

    df = rentals_with_weather.copy()

    df = pd.merge(df, holidays, on="date", how="left")
    df["is_holiday"] = df["holiday"].notna()
    df.drop(columns=["holiday"], inplace=True)

    context.log.info(
        f"Holiday data merged, Final dataset ready. Sample values:\n{df.head()}\nInfo:\n{df.info()}"
    )

    return df

"""Asset: merge holiday information and write the final prepared CSV."""

import pandas as pd
from dagster import asset

from bike_rental.defs.resources.config import DataConfig


@asset(required_resource_keys={"data_config"})
def final_dataset(context, rentals_with_weather: pd.DataFrame) -> pd.DataFrame:
    """Add a holiday flag and produce the final ML-ready dataset.

    The holiday calendar is merged on the ``date`` column produced by
    ``rentals_with_time_features``. A boolean ``is_holiday`` column is
    derived from the presence of a holiday name; the raw name column and
    the helper ``date`` column are then dropped so the persisted dataset
    only carries model-ready features.

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
        holiday columns. Persisted via the configured IO manager (local
        disk or S3).

    """
    cfg: DataConfig = context.resources.data_config

    holidays_path = cfg.get_data_path("holidays.csv")
    holidays = pd.read_csv(
        holidays_path,
        parse_dates=["date"],
        index_col="id",
        storage_options=cfg.storage_options_for(holidays_path),
    )

    df = rentals_with_weather.copy()

    df = pd.merge(df, holidays, on="date", how="left")
    df["is_holiday"] = df["holiday"].notna()
    df.drop(columns=["holiday", "date"], inplace=True)

    context.log.info(
        "Holiday data merged, Final dataset ready. Sample values:\n"
        f"{df.head()}\nInfo:\n{df.info()}"
    )

    return df

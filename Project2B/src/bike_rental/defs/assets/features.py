"""Asset: feature-engineer the prepared dataset for downstream model training.

Mirrors the feature engineering in
``notebooks/03_model_improvement.ipynb`` §2:

- Boolean flags (``is_weekend``, ``is_holiday``) cast to ``int`` so linear
  estimators get a numeric column.
- Rare ``conditions`` categories collapsed (``light_rain`` and
  ``heavy_rain`` → ``rain``); ``heavy_rain`` only has 3 rows in the
  dataset, so its one-hot coefficient would be pure noise otherwise.
- Cyclic encoding of ``hour`` (``hour_sin``, ``hour_cos``) so the model
  sees the 24-hour wrap-around instead of a monotone 0-23 ramp.
- ``is_peak_hour`` interaction flag that explicitly encodes the
  ``hour x is_weekend`` shape diagnosed in the EDA.
- Lag features ``lag_1h``, ``lag_24h``, ``lag_168h`` that capture
  short-term trajectory, daily seasonality, and weekly seasonality +
  YoY growth respectively. The lag shifts introduce NaNs in the first
  168 rows, which are dropped so downstream training sees a complete
  feature matrix.
"""

import numpy as np
import pandas as pd
from dagster import asset


@asset
def engineered_features(
    context, final_dataset: pd.DataFrame
) -> pd.DataFrame:
    """Add cyclic, interaction, and lag features to the prepared dataset.

    Parameters
    ----------
    context : dagster.OpExecutionContext
        Dagster execution context.
    final_dataset : pd.DataFrame
        Output of the ``final_dataset`` asset. Must contain ``datetime``,
        ``total_count``, ``hour``, ``is_weekend``, ``is_holiday``, and
        ``conditions``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame, sorted chronologically, with the engineered
        columns added and the first 168 rows (the lag NaNs) dropped.

    """
    df = final_dataset.copy()
    df = df.sort_values("datetime").reset_index(drop=True)

    df["is_weekend"] = df["is_weekend"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)
    df["conditions"] = df["conditions"].replace(
        {"light_rain": "rain", "heavy_rain": "rain"}
    )

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

    df["is_peak_hour"] = 0
    df.loc[
        (
            (df["hour"].between(7, 9) | df["hour"].between(17, 19))
            & (df["is_weekend"] == 0)
        )
        | (df["hour"].between(12, 16) & (df["is_weekend"] == 1)),
        "is_peak_hour",
    ] = 1

    target = df["total_count"]
    df["lag_1h"] = target.shift(1)
    df["lag_24h"] = target.shift(24)
    df["lag_168h"] = target.shift(168)

    df = df.dropna().reset_index(drop=True)

    context.log.info(
        f"Engineered features ready. Sample values:\n"
        f"{df.head()}\nInfo:\n{df.info()}"
    )

    return df

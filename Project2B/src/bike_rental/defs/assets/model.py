"""Asset: train the XGBoost + log1p regression model and persist it.

The winning configuration from
``notebooks/03_model_improvement.ipynb`` §11:
``TransformedTargetRegressor`` wrapping
``XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.05,
subsample=0.9, colsample_bytree=0.9, tree_method="hist")`` with
``log1p`` / ``expm1`` as the target transform. Test RMSE 45.47, MAE
28.63, R² 0.957 in the notebook.

The fitted estimator is returned from the asset; the configured
``pickle_io_manager`` serializes it to ``<output_dir>/trained_model.pkl``.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from dagster import MetadataValue, asset
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

FEATURES_NUM = [
    "hour",
    "day_of_week",
    "is_weekend",
    "is_holiday",
    "hour_sin",
    "hour_cos",
    "is_peak_hour",
    "lag_1h",
    "lag_168h",
]
FEATURES_CAT = ["conditions"]
FEATURES_ALL = FEATURES_NUM + FEATURES_CAT
TARGET = "total_count"
RANDOM_STATE = 0


@asset(io_manager_key="pickle_io_manager")
def trained_model(
    context, engineered_features: pd.DataFrame
) -> TransformedTargetRegressor:
    """Fit XGBoost + log1p (default) and return the trained pipeline.

    A chronological 80/20 split is used so the holdout reflects the
    deployment setting (predicting future hours from past ones). Train
    and test metrics are logged and attached as asset metadata; the
    fitted estimator is returned and persisted by the configured
    ``pickle_io_manager``.

    Parameters
    ----------
    context : dagster.OpExecutionContext
        Dagster execution context.
    engineered_features : pd.DataFrame
        Output of the ``engineered_features`` asset, sorted chronologically.

    Returns
    -------
    TransformedTargetRegressor
        Fitted estimator that takes the same feature schema used for
        training (``FEATURES_ALL``) and returns predictions in the
        original ``total_count`` scale.

    """
    df = engineered_features

    split_idx = int(len(df) * 0.8)
    df_train = df.iloc[:split_idx].reset_index(drop=True)
    df_test = df.iloc[split_idx:].reset_index(drop=True)

    X_train, y_train = df_train[FEATURES_ALL], df_train[TARGET]
    X_test, y_test = df_test[FEATURES_ALL], df_test[TARGET]

    preprocess = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                FEATURES_CAT,
            ),
        ],
        remainder="passthrough",
    )

    regressor = xgb.XGBRegressor(
        n_estimators=600,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    model = TransformedTargetRegressor(
        regressor=Pipeline(
            [("preprocess", preprocess), ("regressor", regressor)]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        "rmse_train": float(
            np.sqrt(mean_squared_error(y_train, y_pred_train))
        ),
        "rmse_test": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
        "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
        "r2_train": float(r2_score(y_train, y_pred_train)),
        "r2_test": float(r2_score(y_test, y_pred_test)),
    }

    context.log.info(
        f"Trained XGBoost + log1p (default). "
        f"Train: {len(df_train)} rows | Test: {len(df_test)} rows\n"
        f"Metrics: {metrics}"
    )

    context.add_output_metadata(
        {
            "rows_train": len(df_train),
            "rows_test": len(df_test),
            "features": MetadataValue.json(FEATURES_ALL),
            **{k: round(v, 4) for k, v in metrics.items()},
        }
    )

    return model

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

Each training run is also logged to MLflow: parameters, metrics, and the
model artifact go to the configured tracking server, and the resulting
model is registered in the MLflow model registry under
``_REGISTERED_MODEL_NAME``. The freshly registered version
is tagged with the ``candidate`` alias — promotion to ``production`` is
an explicit decision made downstream (see the API in Part 3).
"""

from os import getenv

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import xgboost as xgb
from dagster import MetadataValue, asset
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from bike_rental.defs.resources.lakefs import LakeFSResource
from bike_rental.defs.resources.mlflow import MLflowResource

_MLFLOW_REGISTERED_MODEL_NAME = getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "xgboost-log1p-bike-rental-demand"
)

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

XGB_PARAMS = {
    "n_estimators": 600,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "n_jobs": -1,
    "random_state": 0,
}

CANDIDATE_ALIAS = "candidate"


@asset(
    io_manager_key="pickle_io_manager",
    required_resource_keys={"mlflow_config", "lakefs"},
)
def trained_model(
    context, engineered_features: pd.DataFrame
) -> TransformedTargetRegressor:
    """Fit XGBoost + log1p, log the run to MLflow, register the model.

    A chronological 80/20 split is used so the holdout reflects the
    deployment setting (predicting future hours from past ones). Train
    and test metrics are logged to MLflow and attached as asset metadata;
    the fitted estimator is returned and persisted by the configured
    ``pickle_io_manager``.

    The MLflow run captures:

    - Parameters: XGBoost hyperparameters, feature/target schema, the
      train/test split sizes, and the upstream Dagster run id.
    - Metrics: train/test RMSE, MAE, and R².
    - Artifact: the fitted ``TransformedTargetRegressor`` logged via
      ``mlflow.sklearn`` with an inferred input/output signature.
    - Registry: the model is registered as a new version under
      ``_MLFLOW_REGISTERED_MODEL_NAME`` and tagged with the
      ``candidate`` alias.

    Parameters
    ----------
    context : dagster.OpExecutionContext
        Dagster execution context. ``context.resources.mlflow_config``
        provides the configured ``MLflowResource``.
    engineered_features : pd.DataFrame
        Output of the ``engineered_features`` asset, sorted chronologically.

    Returns
    -------
    TransformedTargetRegressor
        Fitted estimator that takes the same feature schema used for
        training (``FEATURES_ALL``) and returns predictions in the
        original ``total_count`` scale.

    """
    mlflow_cfg: MLflowResource = context.resources.mlflow_config
    mlflow_cfg.configure()
    lakefs_cfg: LakeFSResource = context.resources.lakefs
    data_commit_sha = lakefs_cfg.source_commit_sha()

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

    regressor = xgb.XGBRegressor(**XGB_PARAMS)

    model = TransformedTargetRegressor(
        regressor=Pipeline(
            [("preprocess", preprocess), ("regressor", regressor)]
        ),
        func=np.log1p,
        inverse_func=np.expm1,
    )

    run_name = f"dagster-{context.run_id[:8]}"

    with mlflow.start_run(run_name=run_name) as run:
        tags = {
            "dagster_run_id": context.run_id,
            "dagster_asset": "trained_model",
            "model_family": "xgboost",
            "target_transform": "log1p",
        }
        if data_commit_sha is not None:
            tags["lakefs_repo"] = lakefs_cfg.repo
            tags["lakefs_source_branch"] = lakefs_cfg.source_branch
            tags["lakefs_source_commit"] = data_commit_sha
        mlflow.set_tags(tags)

        params = {
            "model_family": "xgboost",
            **XGB_PARAMS,
            "features_num": FEATURES_NUM,
            "features_cat": FEATURES_CAT,
            "target": TARGET,
            "column_transformer": "onehot_cat",
            "target_transform": "log1p",
            "split_strategy": "chronological_80_20",
            "rows_train": len(df_train),
            "rows_test": len(df_test),
        }
        mlflow.log_params(params)

        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        metrics = {
            "rmse_train": float(
                np.sqrt(mean_squared_error(y_train, y_pred_train))
            ),
            "rmse_test": float(
                np.sqrt(mean_squared_error(y_test, y_pred_test))
            ),
            "mae_train": float(mean_absolute_error(y_train, y_pred_train)),
            "mae_test": float(mean_absolute_error(y_test, y_pred_test)),
            "r2_train": float(r2_score(y_train, y_pred_train)),
            "r2_test": float(r2_score(y_test, y_pred_test)),
        }
        mlflow.log_metrics(metrics)

        signature = infer_signature(X_train, y_pred_train)
        logged_model = mlflow.sklearn.log_model(
            sk_model=model,
            name="xgboost_log1p",
            signature=signature,
            input_example=X_train.head(5),
            registered_model_name=_MLFLOW_REGISTERED_MODEL_NAME,
            pyfunc_predict_fn="predict",
        )

        client = mlflow.MlflowClient()
        latest_version = max(
            client.search_model_versions(
                f"name='{_MLFLOW_REGISTERED_MODEL_NAME}'"
            ),
            key=lambda v: int(v.version),
        )
        client.set_registered_model_alias(
            name=_MLFLOW_REGISTERED_MODEL_NAME,
            alias=CANDIDATE_ALIAS,
            version=latest_version.version,
        )

        context.log.info(
            f"Trained XGBoost + log1p. "
            f"Train: {len(df_train)} rows | Test: {len(df_test)} rows\n"
            f"Metrics: {metrics}\n"
            f"MLflow run_id: {run.info.run_id}\n"
            f"Registered model: {_MLFLOW_REGISTERED_MODEL_NAME} "
            f"v{latest_version.version} (alias='{CANDIDATE_ALIAS}')"
        )

        context.add_output_metadata(
            {
                "rows_train": len(df_train),
                "rows_test": len(df_test),
                "features": MetadataValue.json(FEATURES_ALL),
                "mlflow_run_id": run.info.run_id,
                "mlflow_experiment_id": run.info.experiment_id,
                "mlflow_model_uri": logged_model.model_uri,
                "registered_model": _MLFLOW_REGISTERED_MODEL_NAME,
                "registered_version": int(latest_version.version),
                "registered_alias": CANDIDATE_ALIAS,
                "lakefs_source_commit": data_commit_sha or "n/a",
                **{k: round(v, 4) for k, v in metrics.items()},
            }
        )

    return model

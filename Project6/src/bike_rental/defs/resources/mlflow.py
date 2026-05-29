"""MLflow tracking and model-registry resource.

Encapsulates the tracking-server URI and experiment name so the training
asset stays focused on the modelling logic. The tracking URI follows
MLflow's standard format —
a remote server (``http://...``) or a local store (``file:./mlruns``).

The companion environment variables consumed by ``definitions.py`` are:

- ``MLFLOW_TRACKING_URI``           — tracking server URI.
- ``MLFLOW_EXPERIMENT_NAME``        — experiment runs are grouped under.
"""

from __future__ import annotations

import mlflow
from dagster import ConfigurableResource


class MLflowResource(ConfigurableResource):
    """Configuration for MLflow experiment tracking and registry usage.

    Attributes
    ----------
    tracking_uri : str
        URI of the MLflow tracking server, e.g. ``http://localhost:5000``
        for a hosted server or ``file:./mlruns`` for a local store.
    experiment_name : str
        Name of the MLflow experiment runs are logged under. Created on
        first use.

    """

    tracking_uri: str
    experiment_name: str

    def configure(self) -> None:
        """Point the MLflow client at the configured server + experiment."""
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

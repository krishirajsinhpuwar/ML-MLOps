"""Pickle IO manager for persisting Python objects between Dagster assets.

Mirrors :class:`CSVIOManager` but serializes arbitrary Python objects (typically
fitted scikit-learn / XGBoost estimators) with ``pickle``. Works against either
a local directory or an S3-compatible bucket — fsspec resolves the destination
from the URI scheme.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import fsspec
from dagster import ConfigurableIOManager, InputContext, OutputContext


def _is_remote(location: str) -> bool:
    """Return ``True`` for any fsspec-style URI (``<scheme>://...``)."""
    return "://" in location


class PickleIOManager(ConfigurableIOManager):
    """IO manager that reads and writes Python objects as pickle files.

    Each asset is stored at ``<output_dir>/<asset_key>.pkl``, where
    ``<output_dir>`` is either a local directory or an ``s3://`` URI
    prefix and ``<asset_key>`` joins the Dagster asset-key path with
    double underscores.

    Attributes
    ----------
    output_dir : str
        Destination prefix for asset pickle files.
    storage_options : dict[str, Any] | None
        Optional mapping forwarded to fsspec when reading or writing over
        ``s3://`` URIs. Ignored for local paths.

    """

    output_dir: str
    storage_options: dict[str, Any] | None = None

    def _get_path(self, context: OutputContext | InputContext) -> str:
        """Derive the pickle location for an asset from its asset key."""
        asset_key = "__".join(context.asset_key.path)
        filename = f"{asset_key}.pkl"
        if _is_remote(self.output_dir):
            return f"{self.output_dir.rstrip('/')}/{filename}"
        return str(Path(self.output_dir) / filename)

    def _storage_options_for(self, path: str) -> dict[str, Any]:
        """Only forward `storage_options` for fsspec paths, not local paths."""
        if _is_remote(path) and self.storage_options:
            return self.storage_options
        return {}

    def handle_output(self, context: OutputContext, obj: Any) -> None:
        """Pickle ``obj`` to the asset's destination.

        Parameters
        ----------
        context : OutputContext
            Dagster output context providing the asset key.
        obj : Any
            The Python object produced by the asset (typically a fitted
            estimator).

        """
        path = self._get_path(context)
        if not _is_remote(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        with fsspec.open(path, "wb", **self._storage_options_for(path)) as f:
            pickle.dump(obj, f)
        context.log.info(f"Wrote pickled {type(obj).__name__} to {path}")

    def load_input(self, context: InputContext) -> Any:
        """Unpickle the object written by the upstream asset.

        Parameters
        ----------
        context : InputContext
            Dagster input context providing the upstream asset key.

        Returns
        -------
        Any
            The Python object loaded from disk or object storage.

        """
        path = self._get_path(context)
        with fsspec.open(path, "rb", **self._storage_options_for(path)) as f:
            obj = pickle.load(f)
        context.log.info(f"Read pickled {type(obj).__name__} from {path}")
        return obj

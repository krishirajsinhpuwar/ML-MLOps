"""CSV IO manager for persisting pandas DataFrames between Dagster assets.

Works against either a local directory or an S3-compatible bucket. The
``output_dir`` configuration value is a plain location string — a
filesystem path or an ``s3://`` URI — and pandas is left to resolve it
via fsspec. For S3 targets, ``storage_options`` carries the credentials
and endpoint used by the underlying s3fs client.
"""

from __future__ import annotations
from typing import Any
from pathlib import Path
import pandas as pd
from dagster import ConfigurableIOManager, InputContext, OutputContext

_DATETIME_COLUMNS = ("datetime", "date")


def _is_s3_uri(location: str) -> bool:
    return location.startswith("s3://")


class CSVIOManager(ConfigurableIOManager):
    """IO manager that reads and writes pandas DataFrames as CSV files.

    Each asset is stored at ``<output_dir>/<asset_key>.csv``, where
    ``<output_dir>`` is either a local directory or an ``s3://`` URI
    prefix and ``<asset_key>`` joins the Dagster asset-key path with
    double underscores.

    Attributes
    ----------
    output_dir : str
        Destination prefix for asset CSVs.
    storage_options : dict[str, Any] | None
        Optional mapping forwarded to pandas when reading or writing over
        fsspec-backed URIs. Ignored for local paths.
    """

    output_dir: str
    storage_options: dict[str, Any] | None = None

    def _get_path(self, context: OutputContext | InputContext) -> str:
        """Derive the CSV location for an asset from its asset key."""
        asset_key = "__".join(context.asset_key.path)
        filename = f"{asset_key}.csv"
        if _is_s3_uri(self.output_dir):
            return f"{self.output_dir.rstrip('/')}/{filename}"
        return str(Path(self.output_dir) / filename)

    def _storage_options_for(self, path: str) -> dict[str, Any] | None:
        """Only forward ``storage_options`` for fsspec paths (pandas rejects it otherwise)."""
        return self.storage_options if _is_s3_uri(path) else None

    def handle_output(self, context: OutputContext, obj: pd.DataFrame) -> None:
        """Write a DataFrame to CSV at the asset's destination.

        Parameters
        ----------
        context : OutputContext
            Dagster output context providing the asset key.
        obj : pd.DataFrame
            The DataFrame produced by the asset.
        """
        path = self._get_path(context)
        if not _is_s3_uri(path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        obj.to_csv(path, index=False, storage_options=self._storage_options_for(path))
        context.log.info(f"Wrote {len(obj)} rows to {path}")

    def load_input(self, context: InputContext) -> pd.DataFrame:
        """Read a DataFrame from the CSV written by the upstream asset.

        Parameters
        ----------
        context : InputContext
            Dagster input context providing the upstream asset key.

        Returns
        -------
        pd.DataFrame
            The DataFrame loaded from disk or object storage.
        """
        path = self._get_path(context)
        storage_options = self._storage_options_for(path)
        columns = pd.read_csv(
            path, nrows=0, storage_options=storage_options
        ).columns.tolist()
        parse_cols = [c for c in columns if c in _DATETIME_COLUMNS]
        df = pd.read_csv(
            path,
            parse_dates=parse_cols if parse_cols else False,
            storage_options=storage_options,
        )
        context.log.info(f"Read {len(df)} rows from {path}")
        return df

"""CSV IO manager for persisting pandas DataFrames between Dagster assets."""

from pathlib import Path

import pandas as pd
from dagster import ConfigurableIOManager, InputContext, OutputContext

# Columns that should be parsed as datetimes when loading intermediate CSVs.
# NOTE: ``hour`` is the integer hour-of-day (0–23), not a timestamp — don't parse it.
_DATETIME_COLUMNS = ("datetime", "date")


class CSVIOManager(ConfigurableIOManager):
    """IO manager that reads and writes pandas DataFrames as CSV files.

    Each asset is stored as ``<output_dir>/<asset_key>.csv``.

    Attributes
    ----------
    output_dir : str
        Directory where CSV files are written and read from.
    """

    output_dir: str

    def _get_path(self, context: OutputContext | InputContext) -> Path:
        """Derive the CSV file path for an asset from its asset key.

        Parameters
        ----------
        context : OutputContext | InputContext
            Dagster context carrying the asset key.

        Returns
        -------
        Path
            Full path to the CSV file for this asset.
        """
        asset_key = "__".join(context.asset_key.path)
        return Path(self.output_dir) / f"{asset_key}.csv"

    def handle_output(self, context: OutputContext, obj: pd.DataFrame) -> None:
        """Write a DataFrame to a CSV file.

        Parameters
        ----------
        context : OutputContext
            Dagster output context providing the asset key.
        obj : pd.DataFrame
            The DataFrame produced by the asset.
        """
        path = self._get_path(context)
        path.parent.mkdir(parents=True, exist_ok=True)
        obj.to_csv(path, index=False)
        context.log.info(f"Wrote {len(obj)} rows to {path}")

    def load_input(self, context: InputContext) -> pd.DataFrame:
        """Read a DataFrame from the CSV file written by the upstream asset.

        Parameters
        ----------
        context : InputContext
            Dagster input context providing the upstream asset key.

        Returns
        -------
        pd.DataFrame
            The DataFrame loaded from disk.
        """
        path = self._get_path(context)
        columns = pd.read_csv(path, nrows=0).columns.tolist()
        parse_cols = [c for c in columns if c in _DATETIME_COLUMNS]
        df = pd.read_csv(path, parse_dates=parse_cols if parse_cols else False)
        context.log.info(f"Read {len(df)} rows from {path}")
        return df

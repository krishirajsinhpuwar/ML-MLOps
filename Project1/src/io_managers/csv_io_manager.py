"""CSV IO manager for persisting pandas DataFrames between Dagster assets."""

from pathlib import Path
from dagster import InputContext, IOManager, OutputContext
import pandas as pd


class CSVIOManager(IOManager):
    """IO manager that reads and writes pandas DataFrames as CSV files.

    Each asset is stored as ``<output_dir>/<asset_key>.csv``.  The output
    directory is resolved from the ``DataConfig`` resource attached to the
    run.

    Parameters
    ----------
    output_dir : str
        Directory where CSV files are written and read from.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # internal helper to get the file path for a given asset key
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
        return self.output_dir / f"{asset_key}.csv"

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
        columns = pd.read_csv(path, nrows=0).columns
        parse_cols = ["hour"] if "hour" in columns else []
        df = pd.read_csv(path, parse_dates=parse_cols)
        context.log.info(f"Read {len(df)} rows from {path}")
        return df

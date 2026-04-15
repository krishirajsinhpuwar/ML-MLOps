"""Dagster resources for the bike-sharing preprocessing pipeline."""

from pathlib import Path
from dagster import ConfigurableResource


class DataConfig(ConfigurableResource):
    """Configuration resource that holds file paths for all data sources.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing raw CSV data files.
    output_dir : str
        Path to the directory where processed assets are written.
    """
    
    def __init__(self, data_dir: str, output_dir: str):
        self.data_dir = data_dir
        self.output_dir = output_dir

    def get_data_path(self, filename: str) -> Path:
        """Return an absolute Path for a raw data file.

        Parameters
        ----------
        filename : str
            Name of the file inside ``data_dir``.

        Returns
        -------
        Path
            Absolute path to the requested file.
        """
        return Path(self.data_dir) / filename

    def get_output_path(self, filename: str) -> Path:
        """Return an absolute Path for an output file.

        Parameters
        ----------
        filename : str
            Name of the file inside ``output_dir``.

        Returns
        -------
        Path
            Absolute path to the requested output file.
        """
        path = Path(self.output_dir) / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

"""Dagster resources for the bike-sharing preprocessing pipeline.

``DataConfig`` abstracts over storage location so that the same pipeline
code can read and write either local files or S3 objects (including
S3-compatible stores such as RustFS). Locations are expressed as plain
strings — a filesystem path for local storage, or an ``s3://`` URI for
object storage — and the optional ``storage_options`` mapping is passed
straight through to :func:`pandas.read_csv` / :meth:`DataFrame.to_csv`
when set.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from dagster import ConfigurableResource


def _is_s3_uri(location: str) -> bool:
    return location.startswith("s3://")


def _join(base: str, name: str) -> str:
    """Append ``name`` to ``base``, handling both local paths and S3 URIs."""
    if _is_s3_uri(base):
        return f"{base.rstrip('/')}/{name}"
    return str(Path(base) / name)


class DataConfig(ConfigurableResource):
    """Location configuration for raw inputs and pipeline outputs.

    Attributes
    ----------
    data_dir : str
        Location holding the raw CSV inputs. Either a local directory path
        (e.g. ``"/abs/path/data/raw"``) or an S3 URI prefix
        (e.g. ``"s3://bike-sharing/raw"``).
    output_dir : str
        Location where processed outputs are written. Same format as
        ``data_dir``.
    storage_options : dict[str, Any] | None
        Optional mapping forwarded to pandas when reading or writing over
        fsspec-backed URIs. Required for S3 endpoints that need custom
        credentials or a non-AWS endpoint (RustFS, MinIO, etc.). Ignored
        when the location is a local path.

    """

    data_dir: str
    output_dir: str
    storage_options: dict[str, Any] | None = None

    def get_data_path(self, filename: str) -> str:
        """Return the full location for a raw input file.

        Parameters
        ----------
        filename : str
            Name of the file inside ``data_dir``.

        Returns
        -------
        str
            Local filesystem path or S3 URI pointing at the file.

        """
        return _join(self.data_dir, filename)

    def get_output_path(self, filename: str) -> str:
        """Return the full location for an output file.

        For local destinations the parent directory is created if needed;
        for S3 URIs nothing is created — ``put_object`` handles key
        creation implicitly.

        Parameters
        ----------
        filename : str
            Name of the file inside ``output_dir``.

        Returns
        -------
        str
            Local filesystem path or S3 URI for the requested output.

        """
        location = _join(self.output_dir, filename)
        if not _is_s3_uri(location):
            Path(location).parent.mkdir(parents=True, exist_ok=True)
        return location

    @property
    def is_remote(self) -> bool:
        """Return ``True`` if either the input or output location is remote."""
        return _is_s3_uri(self.data_dir) or _is_s3_uri(self.output_dir)

    def storage_options_for(self, location: str) -> dict[str, Any] | None:
        """Return ``storage_options`` appropriate for ``location``.

        Pandas rejects ``storage_options`` when the target is a plain
        local path, so this returns the configured mapping only for
        fsspec-backed URIs (e.g. ``s3://...``). For local paths it
        returns ``None``. This lets the same config object drive both
        local raw reads and S3 output writes in the mixed-backend setup
        (local raw inputs + S3 outputs).

        Parameters
        ----------
        location : str
            A path or URI previously returned by :meth:`get_data_path`
            or :meth:`get_output_path`.

        """
        return self.storage_options if _is_s3_uri(location) else None

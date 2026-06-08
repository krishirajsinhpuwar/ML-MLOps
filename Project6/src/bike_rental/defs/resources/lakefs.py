"""LakeFS resource for data versioning and asset commits.

Encapsulates the LakeFS endpoint, credentials, repository, and the two
branches used by the pipeline:

- ``source_branch`` (e.g. ``main``) — protected branch the pipeline reads
  raw inputs from. Each training run records the head commit of this
  branch in MLflow so the data version that produced a model is always
  traceable.
- ``output_branch`` (e.g. ``staging``) — branch the pipeline writes
  derived assets (processed CSVs, the model pickle) to. A commit is
  created at the end of each pipeline run with the Dagster run id +
  MLflow run id in the commit metadata, so derived assets are versioned
  alongside the model that produced them. Merging ``staging`` into
  ``main`` is an explicit, gated step (LakeFS branch-protection rules
  configured by ``scripts/lakefs_init.py`` block direct writes to
  ``main``).

When ``enabled`` is ``False`` the resource is a no-op: methods return
``None`` and ``storage_options()`` is empty. This lets the rest of the
pipeline keep one shape regardless of whether ``STORAGE_BACKEND`` is
``lakefs`` or one of the other backends.
"""

from __future__ import annotations

from typing import Any

import lakefs
from dagster import ConfigurableResource
from lakefs.client import Client


class LakeFSResource(ConfigurableResource):
    """Configuration for LakeFS-backed data versioning.

    Attributes
    ----------
    enabled : bool
        Whether this pipeline run is using the LakeFS backend.
    endpoint : str
        LakeFS server URL, e.g. ``http://localhost:8000``.
    access_key : str
        LakeFS access key id.
    secret_key : str
        LakeFS secret access key.
    repo : str
        LakeFS repository name (e.g. ``bike-rental``).
    source_branch : str
        Branch raw inputs are read from. Protected.
    output_branch : str
        Branch derived assets are written to and committed on.

    """

    enabled: bool = False
    storage_options: dict[str, Any] | None = None
    repo: str = "repo"
    source_branch: str = "main"
    output_branch: str = "output"

    def _client(self):
        """Build a lakefs SDK client from the configured credentials.

        ``storage_options`` carries the lakefs-spec ``LakeFSFileSystem``
        connection keys (``host`` / ``username`` / ``password``) — the same
        dict the IO managers forward to fsspec — so the SDK client and the
        filesystem always talk to the same server with the same credentials.
        """
        return Client(
            host=self.storage_options["host"],
            username=self.storage_options["username"],
            password=self.storage_options["password"],
        )

    def _repository(self):
        return lakefs.Repository(self.repo, client=self._client())

    def source_commit_sha(self) -> str | None:
        """Return the head commit SHA of the source branch, or ``None``."""
        if not self.enabled:
            return None
        return self._repository().branch(self.source_branch).head.id

    def commit_output(
        self, message: str, metadata: dict[str, str]
    ) -> str | None:
        """Commit any uncommitted changes on ``output_branch``.

        Parameters
        ----------
        message : str
            Commit message.
        metadata : dict[str, str]
            Commit metadata. Values must be strings.

        Returns
        -------
        str | None
            The new commit SHA, or ``None`` if LakeFS is disabled or there
            was nothing to commit.

        """
        if not self.enabled:
            return None
        branch = self._repository().branch(self.output_branch)
        # ``commit`` is a no-op when there are no uncommitted changes; the
        # SDK either returns the existing head or raises depending on the
        # version, so we capture both shapes by re-reading the head.
        try:
            branch.commit(message=message, metadata=metadata)
        except Exception as exc:  # noqa: BLE001 — surface, don't swallow
            # Empty-commit errors are not fatal; everything else should be.
            if "no changes" not in str(exc).lower():
                raise
        return branch.head.id

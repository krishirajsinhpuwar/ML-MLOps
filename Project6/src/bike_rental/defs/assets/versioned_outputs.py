"""Asset: commit pipeline outputs to LakeFS so derived assets are versioned.

Each Dagster pipeline run writes its derived CSVs and the model pickle to
the LakeFS ``output_branch`` (default ``output``). lakefs-spec stages
those writes as uncommitted changes; this asset turns the staged changes
into a versioned commit and records the commit SHA on the Dagster
materialization. (The matching source-branch commit SHA is recorded
separately on the MLflow run by the ``trained_model`` asset.)

When the active backend is not LakeFS (``STORAGE_BACKEND`` ≠ ``lakefs``)
the resource is in disabled mode and the asset is a no-op: it
materializes successfully so the asset graph stays uniform regardless of
backend.
"""

from __future__ import annotations

from dagster import (
    MaterializeResult,
    MetadataValue,
    asset,
)
from sklearn.compose import TransformedTargetRegressor

from bike_rental.defs.resources.lakefs import LakeFSResource


@asset(required_resource_keys={"lakefs"})
def versioned_outputs(
    context,
    trained_model: TransformedTargetRegressor,
) -> MaterializeResult:
    """Commit the run's derived assets to the LakeFS output branch.

    Taking ``trained_model`` as an input forces this asset to run after
    the model pickle has been written via the ``pickle_io_manager``,
    which is what we want to capture in the commit.
    """
    lakefs_cfg: LakeFSResource = context.resources.lakefs

    if not lakefs_cfg.enabled:
        context.log.info(
            "LakeFS backend disabled — skipping commit "
            "(STORAGE_BACKEND is not 'lakefs')."
        )
        return MaterializeResult(
            metadata={
                "lakefs_enabled": False,
                "note": MetadataValue.text(
                    "LakeFS disabled; derived assets are not versioned."
                ),
            }
        )

    message = f"Dagster run {context.run_id} — pipeline outputs"
    metadata = {"dagster_run_id": context.run_id}
    commit_sha = lakefs_cfg.commit_output(message=message, metadata=metadata)

    context.log.info(
        f"Committed pipeline outputs to "
        f"lakefs://{lakefs_cfg.repo}/{lakefs_cfg.output_branch} "
        f"@ {commit_sha}"
    )

    return MaterializeResult(
        metadata={
            "lakefs_enabled": True,
            "lakefs_repo": lakefs_cfg.repo,
            "lakefs_output_branch": lakefs_cfg.output_branch,
            "lakefs_output_commit": commit_sha or "n/a",
            "commit_message": MetadataValue.text(message),
        }
    )

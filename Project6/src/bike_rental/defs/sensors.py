"""Sensor: retrain automatically when the LakeFS source data changes.

This is the *bonus* of the brief — a fully automated, hands-off pipeline.
``lakefs_source_data_sensor`` polls the head commit of the LakeFS source
branch (the protected ``main`` branch the pipeline reads raw data from)
and launches ``retrain_job`` whenever that commit moves, i.e. whenever new
or corrected raw data lands in the repository.

Why a sensor (and not a schedule)
---------------------------------
The trigger is an *event* ("the data changed"), not a clock. A Dagster
sensor is the matching primitive: it evaluates on a short interval and
emits a ``RunRequest`` only when its observed external state changes.

Change detection & idempotency
-------------------------------
The branch head SHA is the canonical "data changed" signal in a Git-like
store, so the sensor tracks it two ways:

- **Cursor** — the last SHA the sensor acted on is stored in the sensor
  cursor. Each tick compares the current head against it and skips when
  unchanged, so a quiet repository costs one cheap API call per tick.
- **Run key** — each ``RunRequest`` uses the commit SHA as its ``run_key``.
  Dagster launches at most one run per run key for this sensor, so a given
  data version trains exactly once even if the cursor is reset or the
  daemon restarts.

The retrain job reads ``main`` and writes/commits to the ``output`` branch
only, so a run never advances ``main`` — there is no feedback loop.

Enabling
--------
The sensor only does work under the LakeFS backend; on other backends it
emits a skip. It ships **stopped** so ``dg dev`` never launches a training
run unprompted. Turn it on by either:

- setting ``RETRAIN_ON_DATA_CHANGE=true`` in ``.env`` (ships it running), or
- toggling it on in the Dagster UI (Automation → Sensors).
"""

from __future__ import annotations

from os import getenv

from dagster import (
    DefaultSensorStatus,
    RunRequest,
    SensorEvaluationContext,
    SkipReason,
    sensor,
)

from bike_rental.defs.jobs import retrain_job
from bike_rental.defs.resources.lakefs import LakeFSResource

#: How often the daemon evaluates the sensor. 30s is Dagster's typical
#: floor and plenty for a data-versioning trigger.
SENSOR_MIN_INTERVAL_SECONDS = 30


def _default_status() -> DefaultSensorStatus:
    """Ship running only when ``RETRAIN_ON_DATA_CHANGE`` is truthy."""
    flag = getenv("RETRAIN_ON_DATA_CHANGE", "false").strip().lower()
    if flag in {"1", "true", "yes", "on"}:
        return DefaultSensorStatus.RUNNING
    return DefaultSensorStatus.STOPPED


@sensor(
    job=retrain_job,
    minimum_interval_seconds=SENSOR_MIN_INTERVAL_SECONDS,
    default_status=_default_status(),
    description=(
        "Launches retrain_all_assets when the LakeFS source branch head "
        "commit changes (new raw data)."
    ),
)
def lakefs_source_data_sensor(
    context: SensorEvaluationContext, lakefs: LakeFSResource
):
    """Launch a retrain when the source-branch head commit advances.

    Parameters
    ----------
    context : dagster.SensorEvaluationContext
        Provides the persisted ``cursor`` (last SHA acted on) and
        ``update_cursor`` to record the new one.
    lakefs : LakeFSResource
        Injected by key from the Dagster ``Definitions`` resources. Used to
        read the source-branch head commit; a no-op when the backend is not
        LakeFS.

    Yields
    ------
    dagster.RunRequest or dagster.SkipReason
        A ``RunRequest`` (keyed by the new commit SHA) when the head has
        advanced, otherwise a ``SkipReason`` explaining why nothing ran.

    """
    if not lakefs.enabled:
        yield SkipReason(
            "LakeFS backend disabled (STORAGE_BACKEND is not 'lakefs'); "
            "source-data monitoring is inactive."
        )
        return

    try:
        head = lakefs.source_commit_sha()
    except Exception as exc:
        # Surface any connectivity issue as a skip rather than a hard failure.
        yield SkipReason(f"Could not reach LakeFS to read head commit: {exc}")
        return

    if head is None:
        yield SkipReason(
            f"Source branch '{lakefs.source_branch}' has no head commit yet."
        )
        return

    if head == context.cursor:
        yield SkipReason(
            f"No new commit on '{lakefs.source_branch}' "
            f"(head {head[:8]} unchanged)."
        )
        return

    context.update_cursor(head)
    context.log.info(
        f"Detected new commit on '{lakefs.source_branch}': {head[:8]} "
        f"— launching {retrain_job.name}."
    )
    yield RunRequest(
        run_key=head,
        tags={
            "trigger": "lakefs_source_data_sensor",
            "lakefs_source_branch": lakefs.source_branch,
            "lakefs_source_commit": head,
        },
    )

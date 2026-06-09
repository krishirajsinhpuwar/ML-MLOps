"""Jobs that materialize the asset graph.

``retrain_job`` selects every asset in the pipeline (raw CSVs → cleaned
dataset → engineered features → trained model → versioned outputs). It is
the target the auto-retrain sensor (``defs/sensors.py``) launches when the
source data changes, and it can also be run manually from the Dagster UI
or headlessly:

    uv run dagster job execute -j retrain_all_assets -m bike_rental.definitions

Materializing the whole graph (rather than only ``trained_model``) keeps
every run reproducible: the derived CSVs are rebuilt from the current raw
data and re-committed to LakeFS alongside the model that produced them.
"""

from __future__ import annotations

from dagster import AssetSelection, define_asset_job

#: Name reused by the sensor and shown in the Dagster UI.
RETRAIN_JOB_NAME = "retrain_all_assets"

retrain_job = define_asset_job(
    name=RETRAIN_JOB_NAME,
    selection=AssetSelection.all(),
    description=(
        "Rebuild every asset from the current source data and retrain the "
        "model. Launched automatically by lakefs_source_data_sensor when the "
        "LakeFS source branch advances."
    ),
)

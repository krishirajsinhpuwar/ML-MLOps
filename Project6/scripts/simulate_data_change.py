"""Simulate new source data landing in LakeFS, to exercise the bonus sensor.

The auto-retrain sensor (``src/bike_rental/defs/sensors.py``) fires when the
head commit of the protected source branch (``main``) advances. This script
produces exactly that event the *right* way — honouring branch protection.

Because direct writes to ``main`` are blocked, new data cannot be committed
to it directly; it must arrive through a merge. So the script:

1. Downloads a raw trip CSV from the current ``main``.
2. Appends ``--rows`` extra trip records (copies of the last row with fresh
   ids) — a schema-valid change that bumps one hour's rental count, so the
   retrained model genuinely differs.
3. Uploads the modified file to a short-lived ingestion branch and commits.
4. **Merges** the ingestion branch into ``main`` (allowed even though direct
   writes are not) and deletes the ingestion branch.

The result is a new ``main`` head commit. With the sensor enabled and the
Dagster daemon running (``dg dev``), a retrain run starts within one sensor
interval (~30s), and the new model is tagged in MLflow with this commit SHA.

Usage
-----
    uv run python scripts/simulate_data_change.py            # append 1 row
    uv run python scripts/simulate_data_change.py --rows 50  # bigger change
    uv run python scripts/simulate_data_change.py --file weather.csv
"""

from __future__ import annotations

import argparse
import sys
from os import getenv
from pathlib import Path

import lakefs
from dotenv import load_dotenv
from lakefs.client import Client
from lakefs.exceptions import NotFoundException

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(_PROJECT_ROOT / ".env")

_LAKEFS_REPO = getenv("LAKEFS_REPO", "repo")
_LAKEFS_ENDPOINT_URL = getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000")
_LAKEFS_ACCESS_KEY = getenv("LAKEFS_ACCESS_KEY", "admin")
_LAKEFS_SECRET_KEY = getenv("LAKEFS_SECRET_KEY", "admin")
_LAKEFS_SOURCE_BRANCH = getenv("LAKEFS_SOURCE_BRANCH", "main")

_INGEST_BRANCH = "data-ingest"
_DEFAULT_FILE = "direct_pickup_bike_rentals.csv"


def _client() -> Client:
    return Client(
        host=_LAKEFS_ENDPOINT_URL,
        username=_LAKEFS_ACCESS_KEY,
        password=_LAKEFS_SECRET_KEY,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file",
        default=_DEFAULT_FILE,
        help=(
            "Raw CSV under raw/ to append to (default: %(default)s). Trip "
            "files are safest — extra rows just raise an hourly count."
        ),
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=1,
        help="Number of trip rows to append (default: %(default)s).",
    )
    return parser.parse_args()


def _read_raw(repo: lakefs.Repository, key: str) -> str:
    """Return the current text of a raw object on the source branch."""
    obj = repo.branch(_LAKEFS_SOURCE_BRANCH).object(key)
    if not obj.exists():
        raise FileNotFoundError(
            f"{key!r} not found on '{_LAKEFS_SOURCE_BRANCH}'. "
            "Run scripts/lakefs_init.py first."
        )
    with obj.reader(mode="r") as reader:
        return reader.read()


def _append_trip_rows(text: str, n_rows: int) -> str:
    """Append ``n_rows`` copies of the last trip row with fresh ids.

    Trip CSVs are ``id,datetime,user_id,location_id`` with no quoted
    fields, so a plain comma split is safe. Duplicating the final row keeps
    the schema and value ranges valid while genuinely changing the data.
    """
    lines = text.rstrip("\n").split("\n")
    if len(lines) < 2:
        raise ValueError("Raw file has no data rows to copy.")

    last = lines[-1]
    last_id_str, _, remainder = last.partition(",")
    last_id = int(last_id_str)

    new_lines = [f"{last_id + i},{remainder}" for i in range(1, n_rows + 1)]
    return "\n".join(lines + new_lines) + "\n"


def main() -> int:
    """Append rows on an ingestion branch and merge them into ``main``."""
    args = _parse_args()
    key = f"raw/{args.file}"

    client = _client()
    repo = lakefs.Repository(_LAKEFS_REPO, client=client)

    before = repo.branch(_LAKEFS_SOURCE_BRANCH).head.id
    print(f"Source branch '{_LAKEFS_SOURCE_BRANCH}' head: {before}")

    updated = _append_trip_rows(_read_raw(repo, key), args.rows)

    # Fresh ingestion branch off the current source head.
    ingest = repo.branch(_INGEST_BRANCH)
    try:
        ingest.delete()
    except NotFoundException:
        pass
    ingest.create(source_reference=_LAKEFS_SOURCE_BRANCH)
    print(
        f"Created ingestion branch '{_INGEST_BRANCH}' off "
        f"'{_LAKEFS_SOURCE_BRANCH}'."
    )

    ingest.object(key).upload(data=updated.encode(), mode="wb")
    commit = ingest.commit(
        message=f"Add {args.rows} trip record(s) to {args.file}",
        metadata={"source": "scripts/simulate_data_change.py"},
    )
    print(f"Committed change to '{_INGEST_BRANCH}' @ {commit.id}.")

    # Merge is allowed into the protected branch; a direct write is not.
    ingest.merge_into(_LAKEFS_SOURCE_BRANCH)
    ingest.delete()

    after = repo.branch(_LAKEFS_SOURCE_BRANCH).head.id
    print(
        f"Merged into '{_LAKEFS_SOURCE_BRANCH}'. New head: {after}\n"
        f"Appended {args.rows} row(s) to {args.file}."
    )
    if after == before:
        print("Warning: head did not change — was the file already identical?")
        return 1
    print(
        "\nIf lakefs_source_data_sensor is enabled and the Dagster daemon is "
        "running, a retrain run will start within ~30s."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

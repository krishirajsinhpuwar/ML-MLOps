"""Promote a registered model version to the ``production`` alias.

Usage
-----
    # Promote the latest ``candidate`` (default)
    uv run python scripts/promote_model.py

    # Promote a specific version
    uv run python scripts/promote_model.py --version 7

Why a script (and not an asset)
-------------------------------
Promotion is a deliberate decision — every registered version starts as a
``candidate`` (set by the training asset). A human reviews the metrics in
the MLflow UI and runs this script to move the ``production`` alias.
The serving API only ever loads ``models:/<name>@production``, so this
single CLI step is the only thing that controls what users see.
"""

from __future__ import annotations

import argparse
import sys
from os import getenv
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_MLFLOW_REGISTERED_MODEL_NAME = getenv(
    "MLFLOW_REGISTERED_MODEL_NAME", "xgboost-log1p-bike-rental-demand"
)
_MLFLOW_TRACKING_URI = getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
SOURCE_ALIAS = "candidate"
TARGET_ALIAS = "production"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        type=str,
        default=None,
        help=(
            "Specific model version to promote. If omitted, the version "
            f"currently tagged '{SOURCE_ALIAS}' is promoted."
        ),
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=_MLFLOW_REGISTERED_MODEL_NAME,
        help="Registered model name (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> int:
    """Resolve the version to promote and move the ``production`` alias."""
    args = _parse_args()
    mlflow.set_tracking_uri(_MLFLOW_TRACKING_URI)
    client = mlflow.MlflowClient()

    if args.version is not None:
        version = args.version
        source = f"--version {version}"
    else:
        mv = client.get_model_version_by_alias(args.model_name, SOURCE_ALIAS)
        version = mv.version
        source = f"alias '{SOURCE_ALIAS}'"

    client.set_registered_model_alias(
        name=args.model_name, alias=TARGET_ALIAS, version=version
    )
    print(
        f"Promoted {args.model_name} v{version} → alias "
        f"'{TARGET_ALIAS}' (source: {source})."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

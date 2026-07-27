"""Check weather.csv and final_dataset.csv have one row per hour for 2011-2012.

Usage:
    python hour-coverage-issue/check_hourly_coverage.py
    python hour-coverage-issue/check_hourly_coverage.py \
        --weather path/to/weather.csv --final path/to/final_dataset.csv
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEATHER = PROJECT_ROOT / "data" / "raw" / "weather.csv"
DEFAULT_FINAL_LOCAL = (
    PROJECT_ROOT / "data" / "output-local" / "final_dataset.csv"
)

YEARS = (2011, 2012)
EXPECTED_START = pd.Timestamp("2011-01-01 00:00:00")
EXPECTED_END = pd.Timestamp("2012-12-31 23:00:00")


def _load_env(env_path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not env_path.exists():
        return env
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        env[k.strip()] = os.path.expandvars(v.strip())
    # resolve nested vars like ${ENDPOINT_PORT}
    for k, v in list(env.items()):
        for k2, v2 in env.items():
            v = v.replace(f"${{{k2}}}", v2)
        env[k] = v
    return env


def _read_final_dataset(explicit: str | None) -> tuple[pd.DataFrame, str]:
    """Return (df, source). Tries explicit path, then local, then S3."""
    if explicit:
        return pd.read_csv(explicit, parse_dates=["datetime"]), explicit

    if DEFAULT_FINAL_LOCAL.exists():
        return pd.read_csv(DEFAULT_FINAL_LOCAL, parse_dates=["datetime"]), str(
            DEFAULT_FINAL_LOCAL
        )

    env = _load_env(PROJECT_ROOT / ".env")
    bucket = env.get("RUSTFS_BUCKET", "assets")
    endpoint = env.get("RUSTFS_ENDPOINT_URL")
    access_key = env.get("RUSTFS_ACCESS_KEY")
    secret_key = env.get("RUSTFS_SECRET_KEY")
    if not endpoint:
        raise FileNotFoundError(
            f"final_dataset.csv not found at {DEFAULT_FINAL_LOCAL} and no "
            "RUSTFS_ENDPOINT_URL in .env. Pass --final <path> explicitly."
        )
    uri = f"s3://{bucket}/final_dataset.csv"
    storage_options = {
        "key": access_key,
        "secret": secret_key,
        "client_kwargs": {"endpoint_url": endpoint},
    }
    return (
        pd.read_csv(
            uri,
            parse_dates=["datetime"],
            storage_options=storage_options,
        ),
        uri,
    )


def _summarize_gaps(timestamps: pd.Series, label: str) -> None:
    ts = pd.to_datetime(timestamps).sort_values().reset_index(drop=True)
    ts = ts[ts.dt.year.isin(YEARS)]

    expected = pd.date_range(EXPECTED_START, EXPECTED_END, freq="h")
    actual = pd.DatetimeIndex(ts.unique())

    missing = expected.difference(actual)
    duplicates = ts[ts.duplicated()].unique()
    out_of_range = pd.to_datetime(timestamps)[
        ~pd.to_datetime(timestamps).dt.year.isin(YEARS)
    ]

    print(f"\n=== {label} ===")
    print(f"  expected rows : {len(expected):,}")
    print(f"  actual rows   : {len(actual):,}")
    print(f"  missing hours : {len(missing):,}")
    print(f"  duplicates    : {len(duplicates):,}")
    print(f"  out-of-range  : {len(out_of_range):,}")

    if len(missing) == 0 and len(duplicates) == 0:
        print("  OK: complete hourly coverage for 2011-2012.")
        return

    if len(missing):
        print("\n  Missing hours (contiguous gaps):")
        for start, end in _contiguous_ranges(missing):
            if start == end:
                print(f"    - {start} (1 hour)")
            else:
                hours = int((end - start).total_seconds() // 3600) + 1
                print(f"    - {start}  ->  {end}  ({hours} hours)")

    if len(duplicates):
        print("\n  Duplicate hours (first 20):")
        for t in duplicates[:20]:
            print(f"    - {t}")


def _contiguous_ranges(
    idx: pd.DatetimeIndex,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if len(idx) == 0:
        return []
    idx = idx.sort_values()
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    start = prev = idx[0]
    for t in idx[1:]:
        if t - prev == pd.Timedelta(hours=1):
            prev = t
        else:
            gaps.append((start, prev))
            start = prev = t
    gaps.append((start, prev))
    return gaps


def main() -> int:
    """Run the hourly-coverage check against weather and final dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weather", default=str(DEFAULT_WEATHER))
    parser.add_argument(
        "--final", default=None, help="Path or s3:// URI for final_dataset.csv"
    )
    args = parser.parse_args()

    weather = pd.read_csv(args.weather, parse_dates=["datetime"])
    _summarize_gaps(weather["datetime"], f"weather.csv  [{args.weather}]")

    try:
        final_df, final_src = _read_final_dataset(args.final)
    except Exception as exc:
        print(f"\n=== final_dataset.csv ===\n  ERROR: {exc}", file=sys.stderr)
        return 1
    _summarize_gaps(final_df["datetime"], f"final_dataset.csv  [{final_src}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

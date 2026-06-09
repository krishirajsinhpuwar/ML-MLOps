"""Bootstrap the LakeFS repo, branches, raw data, and main-branch protection.

Run once after ``./start-rustfs.sh`` and ``./start-lakefs.sh`` are up.
Idempotent: existing buckets, repos, and branches are reused.

LakeFS stores its objects in RustFS (S3-compatible). The storage
namespace is therefore an ``s3://`` URI inside the same RustFS instance
the rest of the project uses, not a LakeFS-local blockstore.

What this script does
---------------------
1. Ensures the RustFS bucket that backs the LakeFS namespace exists
   (``RUSTFS_BUCKET``, default ``bucket``). Created via s3fs if missing.
2. Creates the LakeFS repository (``LAKEFS_REPO``, default ``repo``) with
   ``main`` as its default branch and the storage namespace pointing
   into the RustFS bucket.
3. Uploads every CSV under ``data/raw/`` to ``main`` and commits.
4. Creates the long-lived output branch off ``main``.
5. Adds a branch-protection rule that blocks direct writes to ``main``
   — updates must come through a merge from the output branch. This is
   the answer to "how do you protect data being merged into main?":
   the pipeline can only ever write to the output branch, and a
   deliberate ``lakectl branch merge`` (or UI action) is required to
   promote it.
"""

from __future__ import annotations

import sys
from os import getenv
from pathlib import Path

import lakefs
import s3fs
from dotenv import load_dotenv
from lakefs.client import Client
from lakefs.exceptions import ConflictException, NotFoundException
from lakefs_sdk.models import BranchProtectionRule

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(_PROJECT_ROOT / ".env")

_RAW_DATA_DIR = Path(getenv("RAW_DATA_DIR", _PROJECT_ROOT / "data/raw"))

_LAKEFS_REPO = getenv("LAKEFS_REPO", "repo")
_LAKEFS_ENDPOINT = getenv("LAKEFS_ENDPOINT_URL", "http://localhost:8000")
_LAKEFS_ACCESS_KEY = getenv("LAKEFS_ACCESS_KEY", "admin")
_LAKEFS_SECRET_KEY = getenv("LAKEFS_SECRET_KEY", "admin")
_LAKEFS_SOURCE_BRANCH = getenv("LAKEFS_SOURCE_BRANCH", "main")
_LAKEFS_OUTPUT_BRANCH = getenv("LAKEFS_OUTPUT_BRANCH", "output")

_RUSTFS_BUCKET = getenv("RUSTFS_BUCKET", "bucket")
_RUSTFS_ENDPOINT_URL = getenv("RUSTFS_ENDPOINT_URL", "http://localhost:9000")
_RUSTFS_ACCESS_KEY = getenv("RUSTFS_ACCESS_KEY", "admin")
_RUSTFS_SECRET_KEY = getenv("RUSTFS_SECRET_KEY", "admin")

# LakeFS stores its objects under this prefix on RustFS. Sharing the
# RustFS bucket with the project's ``STORAGE_BACKEND=s3`` mode is safe:
# the S3 mode writes flat CSVs at the bucket root, LakeFS lives under
# the per-repo prefix.
_LAKEFS_STORAGE_NAMESPACE = getenv(
    "LAKEFS_STORAGE_NAMESPACE", f"s3://{_RUSTFS_BUCKET}/{_LAKEFS_REPO}"
)


def _client() -> Client:
    return Client(
        host=_LAKEFS_ENDPOINT,
        username=_LAKEFS_ACCESS_KEY,
        password=_LAKEFS_SECRET_KEY,
    )


def _ensure_rustfs_bucket() -> None:
    """Create the RustFS bucket LakeFS will write its objects to."""
    fs = s3fs.S3FileSystem(
        key=_RUSTFS_ACCESS_KEY,
        secret=_RUSTFS_SECRET_KEY,
        client_kwargs={"endpoint_url": _RUSTFS_ENDPOINT_URL},
    )
    if fs.exists(_RUSTFS_BUCKET):
        print(f"RustFS bucket {_RUSTFS_BUCKET!r} already exists.")
        return
    fs.mkdir(_RUSTFS_BUCKET)
    print(
        f"Created RustFS bucket {_RUSTFS_BUCKET!r} at {_RUSTFS_ENDPOINT_URL}."
    )


def _ensure_repository(client: Client) -> lakefs.Repository:
    repo = lakefs.Repository(_LAKEFS_REPO, client=client)
    try:
        repo.create(
            storage_namespace=_LAKEFS_STORAGE_NAMESPACE,
            default_branch=_LAKEFS_SOURCE_BRANCH,
            exist_ok=True,
        )
        print(
            f"Repository {_LAKEFS_REPO!r} ready "
            f"(namespace={_LAKEFS_STORAGE_NAMESPACE})."
        )
    except ConflictException:
        print(f"Repository {_LAKEFS_REPO!r} already exists.")
    return repo


def _upload_raw_data(repo: lakefs.Repository) -> None:
    branch = repo.branch(_LAKEFS_SOURCE_BRANCH)
    raw_files = sorted(p for p in _RAW_DATA_DIR.glob("*.csv"))
    if not raw_files:
        print(f"No CSVs under {_RAW_DATA_DIR}; skipping raw data upload.")
        return
    for path in raw_files:
        key = f"raw/{path.name}"
        with path.open("rb") as fh:
            branch.object(key).upload(data=fh.read(), mode="wb")
        print(
            f"Uploaded {path} → lakefs://{_LAKEFS_REPO}/{_LAKEFS_SOURCE_BRANCH}/{key}"
        )
    ref = branch.commit(
        message="Initial raw data load",
        metadata={"source": "scripts/lakefs_init.py"},
    )
    print(
        f"Committed raw data on {_LAKEFS_SOURCE_BRANCH} @ {ref.id} "
        f"({len(raw_files)} files)."
    )


def _ensure_output_branch(repo: lakefs.Repository) -> None:
    try:
        repo.branch(_LAKEFS_OUTPUT_BRANCH).create(
            source_reference=_LAKEFS_SOURCE_BRANCH
        )
        print(
            f"Created branch {_LAKEFS_OUTPUT_BRANCH!r} "
            f"from {_LAKEFS_SOURCE_BRANCH!r}."
        )
    except ConflictException:
        print(f"Branch {_LAKEFS_OUTPUT_BRANCH!r} already exists.")
    except NotFoundException as exc:
        raise RuntimeError(
            f"Source branch {_LAKEFS_SOURCE_BRANCH!r} not found; cannot create "
            f"{_LAKEFS_OUTPUT_BRANCH!r}."
        ) from exc


def _protect_main(client: Client) -> None:
    """Block direct writes to ``main``; only merges may update it."""
    rules = [BranchProtectionRule(pattern=_LAKEFS_SOURCE_BRANCH)]
    try:
        client.sdk_client.repositories_api.set_branch_protection_rules(
            repository=_LAKEFS_REPO, branch_protection_rule=rules
        )
        print(
            f"Branch protection set: direct writes to "
            f"{_LAKEFS_SOURCE_BRANCH!r} are blocked; updates require a "
            f"merge from {_LAKEFS_OUTPUT_BRANCH!r}."
        )
    except Exception as exc:
        print(f"Warning: could not apply branch protection: {exc}")


def main() -> int:
    """Run the full LakeFS bootstrap end to end and print a summary."""
    _ensure_rustfs_bucket()
    client = _client()
    repo = _ensure_repository(client)
    _upload_raw_data(repo)
    _ensure_output_branch(repo)
    _protect_main(client)
    print("\nLakeFS bootstrap complete.")
    print(
        f"  Read branch  : lakefs://{_LAKEFS_REPO}/{_LAKEFS_SOURCE_BRANCH} "
        "(protected)"
    )
    print(f"  Write branch : lakefs://{_LAKEFS_REPO}/{_LAKEFS_OUTPUT_BRANCH}")
    print(f"  Backed by    : {_LAKEFS_STORAGE_NAMESPACE} on RustFS")
    return 0


if __name__ == "__main__":
    sys.exit(main())

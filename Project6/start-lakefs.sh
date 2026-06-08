#!/bin/bash

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "Warning: .env file not found, using default values."
fi

# Set default values using the : ${VAR:=default} syntax
: "${LAKEFS_STORAGE_DIR:=data/output-lakefs}"
: "${LAKEFS_ENDPOINT_PORT:=8000}"
: "${LAKEFS_ACCESS_KEY:=admin}"
: "${LAKEFS_SECRET_KEY:=admin}"

: "${RUSTFS_ENDPOINT_PORT:=9000}"
: "${RUSTFS_ACCESS_KEY:=admin}"
: "${RUSTFS_SECRET_KEY:=admin}"

# From inside the LakeFS container, the host's RustFS port is reached
# via host.docker.internal (Docker Desktop on macOS / Windows).
: "${RUSTFS_HOST_FROM_LAKEFS:=http://host.docker.internal:${RUSTFS_ENDPOINT_PORT}}"

# Ensure directories exist
mkdir -p "$LAKEFS_STORAGE_DIR"

# Stop and remove existing container if it exists (prevents name conflicts)
docker rm -f lakefs-server 2>/dev/null || true

echo "Starting LakeFS on port $LAKEFS_ENDPOINT_PORT (blockstore: RustFS @ $RUSTFS_HOST_FROM_LAKEFS)..."

docker run -d \
  --name lakefs-server \
  -p "${LAKEFS_ENDPOINT_PORT}:8000" \
  -e LAKEFS_DATABASE_TYPE=local \
  -e LAKEFS_DATABASE_LOCAL_PATH=/home/lakefs/data/metadata.db \
  -e LAKEFS_AUTH_ENCRYPT_SECRET_KEY=local-development-secret-please-change \
  -e LAKEFS_BLOCKSTORE_TYPE=s3 \
  -e LAKEFS_BLOCKSTORE_S3_ENDPOINT="${RUSTFS_HOST_FROM_LAKEFS}" \
  -e LAKEFS_BLOCKSTORE_S3_FORCE_PATH_STYLE=true \
  `# Route all object data through the LakeFS server instead of presigned` \
  `# URLs. The blockstore endpoint above (host.docker.internal) is only` \
  `# resolvable from inside the container, so presigned URLs handed to the` \
  `# host-side Python client would be unreachable.` \
  -e LAKEFS_BLOCKSTORE_S3_DISABLE_PRE_SIGNED=true \
  -e LAKEFS_BLOCKSTORE_S3_DISABLE_PRE_SIGNED_UI=true \
  -e LAKEFS_BLOCKSTORE_S3_CREDENTIALS_ACCESS_KEY_ID="${RUSTFS_ACCESS_KEY}" \
  -e LAKEFS_BLOCKSTORE_S3_CREDENTIALS_SECRET_ACCESS_KEY="${RUSTFS_SECRET_KEY}" \
  -e LAKEFS_INSTALLATION_USER_NAME=user \
  -e LAKEFS_INSTALLATION_ACCESS_KEY_ID="${LAKEFS_ACCESS_KEY}" \
  -e LAKEFS_INSTALLATION_SECRET_ACCESS_KEY="${LAKEFS_SECRET_KEY}" \
  -v "$(pwd)/${LAKEFS_STORAGE_DIR}:/home/lakefs/data" \
  treeverse/lakefs:latest \
  run

echo "LakeFS is starting; UI will be available at http://localhost:${LAKEFS_ENDPOINT_PORT}"

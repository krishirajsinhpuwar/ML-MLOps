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
: "${RUSTFS_STORAGE_DIR:=data/output-s3}"
: "${RUSTFS_LOGS_DIR:=logs}"
: "${RUSTFS_ENDPOINT_PORT:=9000}"
: "${RUSTFS_WEB_SERVER_UI_PORT:=9001}"
: "${RUSTFS_ACCESS_KEY:=admin}"
: "${RUSTFS_SECRET_KEY:=admin}"

# Ensure directories exist
mkdir -p "$RUSTFS_STORAGE_DIR"
mkdir -p "$RUSTFS_LOGS_DIR"

# Stop and remove existing container if it exists (prevents name conflicts)
docker rm -f rustfs-server 2>/dev/null || true

echo "Starting RustFS on ports $RUSTFS_ENDPOINT_PORT and $RUSTFS_WEB_SERVER_UI_PORT..."

docker run -d \
  --name rustfs-server \
  -p "${RUSTFS_ENDPOINT_PORT}:9000" \
  -p "${RUSTFS_WEB_SERVER_UI_PORT}:9001" \
  -e "RUSTFS_ACCESS_KEY=${RUSTFS_ACCESS_KEY}" \
  -e "RUSTFS_SECRET_KEY=${RUSTFS_SECRET_KEY}" \
  -v "$(pwd)/${RUSTFS_STORAGE_DIR}:/data" \
  -v "$(pwd)/${RUSTFS_LOGS_DIR}:/logs" \
  rustfs/rustfs:latest

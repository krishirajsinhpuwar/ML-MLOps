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
: "${STORAGE_BACKEND:=local}"
: "${LOGS_DIR:=logs}"
: "${MLFLOW_ENDPOINT_PORT:=5000}"
: "${MLFLOW_BACKEND_STORE_URI:=sqlite:///mlflow.db}"

# Ensure directories exist
mkdir -p $LOGS_DIR

case "${STORAGE_BACKEND}" in
    lakefs)
        : "${LAKEFS_REPO:=repo}"
        : "${LAKEFS_OUTPUT_BRANCH:=output}"
        : "${LAKEFS_ENDPOINT_URL:=http://localhost:8000}"
        : "${LAKEFS_ACCESS_KEY:=admin}"
        : "${LAKEFS_SECRET_KEY:=admin}"
        : "${AWS_DEFAULT_REGION:=us-east-1}"

        export MLFLOW_S3_ENDPOINT_URL="${LAKEFS_ENDPOINT_URL}"
        export AWS_ACCESS_KEY_ID="${LAKEFS_ACCESS_KEY}"
        export AWS_SECRET_ACCESS_KEY="${LAKEFS_SECRET_KEY}"
        export AWS_DEFAULT_REGION

        MLFLOW_ARTIFACT_STORE_URI="s3://${LAKEFS_REPO}/${LAKEFS_OUTPUT_BRANCH}/mlartifacts"

        STORAGE_DESCRIPTION="LakeFS (${LAKEFS_ENDPOINT_URL})"
        ;;

    s3)
        : "${S3_BUCKET:=${RUSTFS_BUCKET:-bucket}}"
        : "${S3_ENDPOINT_URL:=${RUSTFS_ENDPOINT_URL:-http://localhost:9000}}"
        : "${S3_ACCESS_KEY:=${RUSTFS_ACCESS_KEY:-admin}}"
        : "${S3_SECRET_KEY:=${RUSTFS_SECRET_KEY:-admin}}"
        : "${AWS_DEFAULT_REGION:=us-east-1}"

        export MLFLOW_S3_ENDPOINT_URL="${S3_ENDPOINT_URL}"
        export AWS_ACCESS_KEY_ID="${S3_ACCESS_KEY}"
        export AWS_SECRET_ACCESS_KEY="${S3_SECRET_KEY}"
        export AWS_DEFAULT_REGION

        MLFLOW_ARTIFACT_STORE_URI="s3://${S3_BUCKET}/mlartifacts"

        STORAGE_DESCRIPTION="S3 (${S3_ENDPOINT_URL})"
        ;;

    local)
        ARTIFACT_DIR="${MLFLOW_ARTIFACT_DIR:-$(pwd)/mlartifacts}"
        mkdir -p "${ARTIFACT_DIR}"

        MLFLOW_ARTIFACT_STORE_URI="file://${ARTIFACT_DIR}"

        STORAGE_DESCRIPTION="Local filesystem (${ARTIFACT_DIR})"
        ;;
esac

echo "Starting MLflow tracking server"
echo "  Port          : ${MLFLOW_ENDPOINT_PORT}"
echo "  Backend Store : ${MLFLOW_BACKEND_STORE_URI}"
echo "  Artifact Store: ${MLFLOW_ARTIFACT_STORE_URI}"
echo "  Storage Type  : ${STORAGE_DESCRIPTION}"

exec uv run mlflow server \
    --backend-store-uri "${MLFLOW_BACKEND_STORE_URI}" \
    --artifacts-destination "${MLFLOW_ARTIFACT_STORE_URI}" \
    --host 0.0.0.0 \
    --port "${MLFLOW_ENDPOINT_PORT}" \
    > $LOGS_DIR/mlflow.log 2>&1

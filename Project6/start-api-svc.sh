#!/bin/bash
#
# Start the FastAPI prediction service (uvicorn) on :8800. It loads the
# model carrying the 'production' alias from the MLflow registry, so the
# MLflow server must be reachable. Logs to $LOGS_DIR/api.log.

# Load environment variables from .env file if it exists
if [ -f .env ]; then
    set -a
    source .env
    set +a
else
    echo "Warning: .env file not found, using default values."
fi

# Set default values using the : ${VAR:=default} syntax
: "${LOGS_DIR:=logs}"

# Ensure directories exist
mkdir -p $LOGS_DIR

uv run uvicorn api.main:app \
    --host 0.0.0.0 \
    --port 8800 \
    > $LOGS_DIR/api.log 2>&1

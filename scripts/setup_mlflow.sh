#!/bin/bash

# Exit on any error
set -e

# Default MLflow parameters (can override with env variables)
MLFLOW_PORT=${MLFLOW_PORT:-5000}
MLFLOW_ARTIFACT_ROOT=${MLFLOW_ARTIFACT_ROOT:-./mlruns}
MLFLOW_BACKEND_STORE_URI=${MLFLOW_BACKEND_STORE_URI:-sqlite:///mlflow.db}

echo "Starting MLflow server..."
echo "Backend store URI: $MLFLOW_BACKEND_STORE_URI"
echo "Artifact root: $MLFLOW_ARTIFACT_ROOT"
echo "Port: $MLFLOW_PORT"

# Create mlruns folder if it doesn't exist
mkdir -p $MLFLOW_ARTIFACT_ROOT

# Start MLflow server
mlflow server \
    --backend-store-uri $MLFLOW_BACKEND_STORE_URI \
    --default-artifact-root $MLFLOW_ARTIFACT_ROOT \
    --host 0.0.0.0 \
    --port $MLFLOW_PORT

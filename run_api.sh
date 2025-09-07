#!/bin/bash
# run_api.sh - Script to start the FastAPI service

echo "Starting RadX-CV Medical AI API Service..."

# Step 1: Build the Docker image (optional)
if [ "$1" == "build" ]; then
    echo "Building Docker image..."
    docker build -t radx-cv:api .
fi

# Step 2: Run the container with GPU support
echo "Starting API service on port 8000..."

# Path to your Google service account JSON file
GCP_SA_FILE="$(pwd)/firestore-access.json"

docker run --rm --gpus all \
    -p 8000:8000 \
    -v "$(pwd)":/workspace \
    -v "$(pwd)/.env":/workspace/.env \
    -v "$(pwd)/models":/app/models \
    -v "$(pwd)/data":/app/data \
    -v "$(pwd)/outputs":/app/outputs \
    -v "$(pwd)/rag":/app/rag \
    -v "$(pwd)/api":/app/api \
    -v "$GCP_SA_FILE":/workspace/firestore-access.json \
    -w /workspace \
    -e GOOGLE_APPLICATION_CREDENTIALS=/workspace/firestore-access.json \
    radx-cv:api \
    bash -c "\
        echo 'Installing requirements...'; \
        pip install --no-cache-dir -r requirements.txt && \
        echo 'Starting FastAPI service...'; \
        uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload \
    "

echo "API service stopped."

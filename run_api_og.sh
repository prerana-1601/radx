#!/bin/bash
# run_api.sh - Script to start the FastAPI service

echo "Starting RadX-CV Medical AI API Service..."

# Build the Docker image - t build # ./run_api.sh build

echo "Building Docker image..."
if [ "$1" == "build" ]; then
    echo "Building Docker image..."
    docker build -t radx-cv:api .
fi
#docker build -t radx-cv:api .

# Run the container with GPU support
echo "Starting API service on port 8000..."
docker run --rm --gpus all \
    -p 8000:8000 \
    -v $(pwd):/workspace \
    -v $(pwd)/.env:/workspace/.env \
    -v $(pwd)/models:/app/models \
    -v $(pwd)/data:/app/data \
    -v $(pwd)/outputs:/app/outputs \
    -v $(pwd)/rag:/app/rag \
    -v $(pwd)/api:/app/api \
    -w /workspace \
    radx-cv:api \
    uvicorn api_main:app --host 0.0.0.0 --port 8000 --reload
    #unicorn looks at api_main.py file and app the instance of the FAST API
    #when unicorn loads the python file it registers all the routes @app.get, app@posst, app.on_event
    #once all routes are registered it calls @app.on_event("startup")
    #after starup it starts listening at http://0.0.0.0:8000
echo "API service stopped."
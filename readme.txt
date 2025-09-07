- Creating a folder
mkdir ~/radx_project
cd ~/radx_project

- Creating subdirectories
mkdir data models notebooks api rag

- Download data from - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia?resource=download

- Unzip into data subdirectory

-Populate requirements.txt

-Populate Dockerfile

-Populate train.py

- Building the docker container
docker build -t radx-cv:latest . 

- To Run the container
docker run --rm -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models radx-cv:latest
--rm - removes container after it stops
-v $(pwd)/data:/app/data - mounts dataset
-v $(pwd)/models:/app/models - saves model checkpoints
radx-cv:latest - runs the image


enter docker and check cuda compatibility


--------------------------------------------------------------------------

Workflow Orchestration

Airflow/Prefect for scheduled jobs.

DBT for data transformations if structured data is involved.

API Layer

FastAPI to expose model endpoints.

Authentication (JWT/OAuth).

Deployment

Dockerize pipeline.

Deploy on AWS (ECS/Lambda/SageMaker) or cheaper alternative (Render, Railway).

Frontend (optional but impressive)

Simple Streamlit/React dashboard where user uploads file and sees results.



-----------
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
    radx-cv:api

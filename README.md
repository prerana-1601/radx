# RADx Project

This project provides a **computer vision–based API** and **Streamlit interface** for analyzing medical images using deep learning models.  
It includes Dockerized services for easy deployment and uses Google Cloud Firestore for storing uploaded X-rays and analysis results.

---

## Running the Project

### 1. Clone the repository
```bash
git clone https://github.com/prerana-1601/radx.git
cd radx
```

### 2. Create a .env file

Add your environment variables (e.g., your OpenAI API key).

```bash
OPENAI_API_KEY=your_api_key_here
```

### 3. Firestore Credentials

The project requires a Google Cloud Firestore service account key (firestore-access.json) for storing uploaded X-rays and analysis results.

The JSON file should follow this format (example values, not real):

```bash
{
  "type": "service_account",
  "project_id": "your-project-id",
  "private_key_id": "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx",
  "private_key": "-----BEGIN PRIVATE KEY-----\n...\n-----END PRIVATE KEY-----\n",
  "client_email": "your-service-account@your-project-id.iam.gserviceaccount.com",
  "client_id": "xxxxxxxxxxxxxxxxxxxx",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/your-service-account%40your-project-id.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}
```

#### Instructions to create your own

1. Go to Google Cloud Console → IAM & Admin → Service Accounts

2. Create a new service account.

3. Assign the Firestore User role (or a custom role with read/write to Firestore).

4. Generate a new JSON key and download it as firestore-access.json.

5. Place the file in the project root and set the environment variable:

```bash
export GOOGLE_APPLICATION_CREDENTIALS=firestore-access.json
```

### 4. Build and run with Docker Compose

```bash
docker compose up --build
```

### This will spin up:

->API service (FastAPI + Uvicorn, available at http://localhost:8000)

>UI service (Streamlit app, available at http://localhost:8501)



### 5. Project Structure

```text
radx_project/
├── api/                     # API code
├── streamlit/               # Streamlit frontend
├── models/                  # Trained models
├── data/                    # Input data
├── outputs/                 # Results / logs
├── Dockerfile               # API Docker build
├── docker-compose.yaml      # Multi-container setup
├── requirements.txt         # Python dependencies
├── run_api.sh               # Script to run API
├── firestore-access.json    # (excluded from Git)
└── .env                     # Environment variables (excluded from Git)
```
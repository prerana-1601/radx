# api_main.py - FastAPI service integrated with Firestore persistence
import os
import uuid
import json
import shutil
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

from google.cloud import firestore  # Firestore client

# Import your existing modules
from rag.llm_chain import get_qa_chain, ask_qa
from rag.schemas import PneumoniaInfo
from api.gradcam import GradCAM

# -----------------------------
# App & storage configuration
# -----------------------------
app = FastAPI(
    title="RadX-CV Medical AI System",
    description="End-to-end X-ray analysis with AI-powered reporting and GradCAM visualization",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local storage paths (ensure these map correctly in your Docker binds)
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")
GRADCAM_DIR = Path("outputs/gradcam")
REPORT_DIR = Path("outputs/reports")

for dir_path in [UPLOAD_DIR, RESULTS_DIR, GRADCAM_DIR, REPORT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Firestore setup
# -----------------------------
db = firestore.Client()  # uses GOOGLE_APPLICATION_CREDENTIALS env var
COLLECTION = "analysis_store"  # Firestore collection name (auto-created)

# -----------------------------
# Globals (model, transforms)
# -----------------------------
device = None
model = None
gradcam = None
transform = None
qa_chain = None
classes = ["NORMAL", "PNEUMONIA"]

class AnalysisStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

# -----------------------------
# Pydantic models
# -----------------------------
class AnalysisRequest(BaseModel):
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    study_description: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str
    created_at: str

# -----------------------------
# Firestore helper functions
# -----------------------------
def save_record(analysis_id: str, record: dict):
    """Create or overwrite a record in Firestore."""
    # Firestore stores datetimes as string for portability here
    db.collection(COLLECTION).document(analysis_id).set(record)

def get_record(analysis_id: str) -> Optional[dict]:
    """Get a record (returns dict or None)."""
    doc = db.collection(COLLECTION).document(analysis_id).get()
    return doc.to_dict() if doc.exists else None

def update_record(analysis_id: str, updates: dict):
    """Atomic update of a record in Firestore."""
    # Firestore expects a mapping of fields to values
    db.collection(COLLECTION).document(analysis_id).update(updates)

def delete_record(analysis_id: str):
    db.collection(COLLECTION).document(analysis_id).delete()

def list_records(limit: int = 10, offset: int = 0) -> List[dict]:
    """List recent records. For small data, offset is implemented client-side if necessary."""
    # We store ISO timestamps in created_at; ordering by that string works for ISO.
    query = db.collection(COLLECTION).order_by("created_at", direction=firestore.Query.DESCENDING).limit(limit + offset)
    docs = query.stream()
    items = [d.to_dict() for d in docs]
    # Apply offset client-side (Firestore supports start_after / cursors for production)
    return items[offset:offset + limit]

# -----------------------------
# Startup: load model, gradcam, transforms, QA chain
# -----------------------------
@app.on_event("startup")
async def startup_event():
    global device, model, gradcam, transform, qa_chain

    print("Starting RadX-CV Medical AI System...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading ResNet18 model...")
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)

    model_path = MODEL_DIR / "resnet18_xray.pth"
    if model_path.exists():
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    else:
        print("Model file not found, using random weights")

    model = model.to(device)
    model.eval()

    # Setup GradCAM
    target_layer = model.layer4[1].conv2
    gradcam = GradCAM(model, target_layer)
    print("GradCAM initialized")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # QA chain
    try:
        qa_chain = get_qa_chain(temperature=0.0)
        print("LLM QA chain initialized")
    except Exception as e:
        print(f"Could not initialize QA chain: {e}")
        qa_chain = None

    print("System startup complete!")

# -----------------------------
# Domain functions (same logic, adapted to Firestore)
# -----------------------------
def analyze_gradcam_regions(heatmap, pred_class):
    height, width = heatmap.shape
    max_activation = float(np.max(heatmap))
    threshold = max_activation * 0.7

    activated_mask = heatmap > threshold
    regions = []

    upper_third = height // 3
    lower_third = 2 * height // 3
    left_half = width // 2

    if np.any(activated_mask[:upper_third, :]):
        if np.any(activated_mask[:upper_third, :left_half]):
            regions.append("left upper lung field")
        if np.any(activated_mask[:upper_third, left_half:]):
            regions.append("right upper lung field")

    if np.any(activated_mask[upper_third:lower_third, :]):
        if np.any(activated_mask[upper_third:lower_third, :left_half]):
            regions.append("left middle lung field")
        if np.any(activated_mask[upper_third:lower_third, left_half:]):
            regions.append("right middle lung field")

    if np.any(activated_mask[lower_third:, :]):
        if np.any(activated_mask[lower_third:, :left_half]):
            regions.append("left lower lung field")
        if np.any(activated_mask[lower_third:, left_half:]):
            regions.append("right lower lung field")

    if pred_class == 1:  # PNEUMONIA
        if regions:
            region_text = ", ".join(regions)
            description = f"Chest X-ray shows abnormal opacities in the {region_text}. AI model detected features consistent with pneumonia with high confidence. The areas of concern show increased density patterns typical of inflammatory processes."
        else:
            description = "Chest X-ray shows features suggestive of pneumonia. AI analysis indicates abnormal lung parenchyma consistent with infectious process."
    else:
        description = "Chest X-ray appears within normal limits. Lung fields are clear with no evidence of acute cardiopulmonary abnormalities. AI analysis supports normal chest radiograph findings."

    return description, regions

def generate_comprehensive_report(image_analysis, pred_class, confidence, regions):
    if qa_chain is None:
        if pred_class == 1:
            findings = "Abnormal chest radiograph findings consistent with pneumonic consolidation."
            impression = "Findings consistent with pneumonia."
            recommendations = "Clinical correlation recommended. Consider antibiotic therapy if clinically indicated."
        else:
            findings = "Normal chest radiograph. Clear lung fields bilaterally."
            impression = "No acute cardiopulmonary abnormalities."
            recommendations = "Routine follow-up as clinically indicated."
        return findings, impression, recommendations

    findings_prompt = f"""
    Based on this chest X-ray analysis: {image_analysis}
    Generate detailed radiological findings in proper medical terminology.
    Include specific observations about lung fields, heart size, and any abnormalities noted.
    Keep it concise and professional.
    """
    findings = ask_qa(findings_prompt, qa_chain)

    if pred_class == 1:
        impression_prompt = f"""
        Given these chest X-ray findings showing pneumonia in regions: {', '.join(regions) if regions else 'multiple areas'}
        Provide a clinical impression statement that would be appropriate for a radiologist's report.
        Include the likely diagnosis and any relevant clinical correlations needed.
        """
    else:
        impression_prompt = """
        For a normal chest X-ray with clear lung fields and no abnormalities,
        provide a standard radiological impression statement.
        """

    impression = ask_qa(impression_prompt, qa_chain)

    if pred_class == 1:
        recommendations_prompt = """
        For a chest X-ray showing findings consistent with pneumonia,
        what clinical recommendations should a radiologist include?
        Focus on follow-up imaging, clinical correlation, and next steps.
        """
        recommendations = ask_qa(recommendations_prompt, qa_chain)
    else:
        recommendations = "No acute findings. Routine clinical correlation as indicated."

    return findings, impression, recommendations

def get_additional_info(condition):
    if qa_chain is None or condition == "NORMAL":
        if condition == "PNEUMONIA":
            return (
                "Common causes include bacterial, viral, or fungal infections.",
                "Treatment typically involves antibiotics for bacterial pneumonia, supportive care for viral.",
                "Prevention includes vaccination, hand hygiene, and avoiding smoking."
            )
        else:
            return (
                "No pathological condition detected.",
                "No treatment required for normal chest X-ray findings.",
                "Maintain good respiratory health through regular exercise, vaccination, and avoiding smoking."
            )

    if condition == "PNEUMONIA":
        causes_prompt = "What are the main causes of pneumonia? Provide a concise medical overview."
        treatments_prompt = "What are the standard treatments for pneumonia? Include both outpatient and inpatient approaches."
        prevention_prompt = "How can pneumonia be prevented? List key prevention strategies."

        causes = ask_qa(causes_prompt, qa_chain)
        treatments = ask_qa(treatments_prompt, qa_chain)
        prevention = ask_qa(prevention_prompt, qa_chain)
        return causes, treatments, prevention
    else:
        return get_additional_info("NORMAL")

# -----------------------------
# Background processing (uses Firestore)
# -----------------------------
async def process_xray_analysis(analysis_id: str):
    try:
        # Get record from Firestore
        record = get_record(analysis_id)
        if not record:
            print(f"[process_xray] Record {analysis_id} not found.")
            return

        # Update status -> processing
        update_record(analysis_id, {"status": AnalysisStatus.PROCESSING, "updated_at": datetime.now().isoformat()})

        file_path = record.get("file_path")
        filename = record.get("filename")
        if not file_path or not os.path.exists(file_path):
            update_record(analysis_id, {"status": AnalysisStatus.FAILED, "updated_at": datetime.now().isoformat(), "error": "Uploaded file not found"})
            return

        # Load and process image
        img = Image.open(file_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Forward pass
        output = model(input_tensor)
        pred_class = int(torch.argmax(output, 1).item())
        confidence = float(torch.softmax(output, 1)[0, pred_class].item())

        # Backprop for GradCAM
        score = output[0, pred_class]
        model.zero_grad()
        score.backward()

        # Generate GradCAM heatmap
        heatmap = gradcam.generate_heatmap(pred_class)
        heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        img_np = np.array(img)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

        # Save gradcam overlay
        gradcam_filename = f"API_{analysis_id}_{filename}_gradcam.jpg"
        gradcam_path = GRADCAM_DIR / gradcam_filename
        cv2.imwrite(str(gradcam_path), overlay)

        # Analyze regions and generate report text
        image_analysis, regions = analyze_gradcam_regions(heatmap, pred_class)
        findings, impression, recommendations = generate_comprehensive_report(image_analysis, pred_class, confidence, regions)
        causes, treatments, prevention = get_additional_info(classes[pred_class])

        # Build report JSON (PneumoniaInfo)
        prediction_summary = f"""Model Classification: {classes[pred_class]}
Confidence Score: {confidence:.3f} ({confidence*100:.1f}%)
Analysis Method: ResNet18 CNN with GradCAM visualization
Affected Regions: {', '.join(regions) if regions else 'Diffuse lung fields'}
Image: {filename}"""

        full_report = f"""Findings:
{findings}
- Highlighted regions on Grad-CAM show attention focused on: {', '.join(regions) if regions else 'diffuse lung fields'}
- AI model prediction: {classes[pred_class]} (confidence: {confidence:.2f})

Impression:
{impression}

Recommendations:
{recommendations}"""

        report_obj = PneumoniaInfo(
            prediction=prediction_summary.strip(),
            findings=full_report.strip(),
            causes=causes,
            treatments=treatments,
            prevention=prevention
        )

        # Save report JSON to disk
        report_filename = f"API_{analysis_id}_{filename}_report.json"
        report_path = REPORT_DIR / report_filename
        with open(report_path, "w") as f:
            f.write(report_obj.model_dump_json(indent=2))

        # Update Firestore record with results
        update_record(analysis_id, {
            "status": AnalysisStatus.COMPLETED,
            "updated_at": datetime.now().isoformat(),
            "prediction": classes[pred_class],
            "confidence": confidence,
            "regions": regions,
            "report_path": str(report_path),
            "gradcam_path": str(gradcam_path),
            "report_data": report_obj.model_dump(),
            "processing_time": (datetime.now() - datetime.fromisoformat(record["created_at"])).total_seconds() if record.get("created_at") else None
        })

        print(f"Analysis {analysis_id} completed: {classes[pred_class]} ({confidence:.2f})")

    except Exception as e:
        print(f"Analysis {analysis_id} failed: {str(e)}")
        try:
            update_record(analysis_id, {"status": AnalysisStatus.FAILED, "updated_at": datetime.now().isoformat(), "error": str(e)})
        except Exception:
            pass

# -----------------------------
# API endpoints (use Firestore)
# -----------------------------
@app.post("/api/analyze-xray", response_model=AnalysisResponse)
async def analyze_xray(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None
):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")

    analysis_id = str(uuid.uuid4())

    # Save uploaded file
    file_path = UPLOAD_DIR / f"{analysis_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Create record and save to Firestore
    now_iso = datetime.now().isoformat()
    analysis_record = {
        "id": analysis_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "patient_id": patient_id,
        "patient_name": patient_name,
        "status": AnalysisStatus.PENDING,
        "created_at": now_iso,
        "updated_at": now_iso
    }

    save_record(analysis_id, analysis_record)

    # Queue background processing (async function)
    background_tasks.add_task(process_xray_analysis, analysis_id)

    return AnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.PENDING,
        message="Analysis queued for processing",
        created_at=now_iso
    )

@app.get("/api/analysis/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    record = get_record(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    response = {
        "id": analysis_id,
        "status": record.get("status"),
        "created_at": record.get("created_at"),
        "updated_at": record.get("updated_at"),
        "filename": record.get("filename"),
        "patient_info": {
            "id": record.get("patient_id"),
            "name": record.get("patient_name")
        }
    }

    if record.get("status") == AnalysisStatus.COMPLETED:
        response.update({
            "prediction": record.get("prediction"),
            "confidence": record.get("confidence"),
            "regions": record.get("regions"),
            "processing_time": record.get("processing_time"),
            "gradcam_url": f"/api/analysis/{analysis_id}/gradcam",
            "report_url": f"/api/analysis/{analysis_id}/report"
        })
    elif record.get("status") == AnalysisStatus.FAILED:
        response["error"] = record.get("error")

    return response

@app.get("/api/analysis/{analysis_id}/report")
async def get_analysis_report(analysis_id: str):
    record = get_record(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if record.get("status") != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Analysis not completed. Status: {record.get('status')}")

    report_path = record.get("report_path")
    if report_path and os.path.exists(report_path):
        return FileResponse(report_path, media_type="application/json")
    raise HTTPException(status_code=404, detail="Report not found")

@app.get("/api/analysis/{analysis_id}/gradcam")
async def get_gradcam_image(analysis_id: str):
    record = get_record(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    if record.get("status") != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Analysis not completed. Status: {record.get('status')}")

    gradcam_path = record.get("gradcam_path")
    if gradcam_path and os.path.exists(gradcam_path):
        return FileResponse(gradcam_path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="GradCAM image not found")

@app.get("/api/analyses")
async def list_analyses(limit: int = 10, offset: int = 0):
    records = list_records(limit=limit, offset=offset)
    total = len(records)  # not the global total; for full total consider a count aggregation
    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "analyses": [
            {
                "id": r.get("id"),
                "filename": r.get("filename"),
                "status": r.get("status"),
                "created_at": r.get("created_at"),
                "patient_name": r.get("patient_name"),
                "prediction": r.get("prediction"),
                "confidence": r.get("confidence")
            } for r in records
        ]
    }

@app.delete("/api/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    record = get_record(analysis_id)
    if not record:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Delete files on disk if present
    try:
        file_path = record.get("file_path")
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

        gradcam_path = record.get("gradcam_path")
        if gradcam_path and os.path.exists(gradcam_path):
            os.remove(gradcam_path)

        report_path = record.get("report_path")
        if report_path and os.path.exists(report_path):
            os.remove(report_path)
    except Exception as e:
        print(f"Error deleting files for {analysis_id}: {e}")

    # Delete Firestore record
    delete_record(analysis_id)
    return {"message": "Analysis deleted successfully"}

@app.get("/api/health")
async def health_check():
    # small health info; avoid expensive counts in production
    # Try to fetch a few docs for a quick health check
    try:
        docs = db.collection(COLLECTION).limit(1).stream()
        total_sample = sum(1 for _ in docs)
    except Exception:
        total_sample = 0

    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "qa_chain_available": qa_chain is not None,
        "total_analyses_sample": total_sample
    }

@app.get("/")
async def root():
    return {
        "service": "RadX-CV Medical AI System",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "description": "End-to-end X-ray analysis with AI-powered reporting"
    }

if __name__ == "__main__":
    uvicorn.run(
        "api_main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1
    )

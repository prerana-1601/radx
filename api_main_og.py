# api_main.py - FastAPI service integrated with your existing pipeline
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import json
import uuid
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional, List
import shutil

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import uvicorn

# Import your existing modules
from rag.llm_chain import get_qa_chain, ask_qa
from rag.schemas import PneumoniaInfo
from api.gradcam import GradCAM

# FastAPI app setup
app = FastAPI(
    title="RadX-CV Medical AI System",
    description="End-to-end X-ray analysis with AI-powered reporting and GradCAM visualization",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Storage directories - match your existing structure
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
MODEL_DIR = Path("models")
GRADCAM_DIR = Path("outputs/gradcam")
REPORT_DIR = Path("outputs/reports")

# Create directories
for dir_path in [UPLOAD_DIR, RESULTS_DIR, GRADCAM_DIR, REPORT_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Global variables for model (loaded once at startup)
device = None
model = None
gradcam = None
transform = None
qa_chain = None
classes = ["NORMAL", "PNEUMONIA"]

# In-memory storage (replace with database in production)
analysis_store = {}

class AnalysisStatus:
    PENDING = "pending"
    PROCESSING = "processing"  
    COMPLETED = "completed"
    FAILED = "failed"

# Pydantic models for API
class AnalysisRequest(BaseModel):
    patient_id: Optional[str] = None
    patient_name: Optional[str] = None
    study_description: Optional[str] = None

class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    message: str
    created_at: str

@app.on_event("startup")
async def startup_event():
    """Initialize model and components on startup"""
    global device, model, gradcam, transform, qa_chain
    
    print("Starting RadX-CV Medical AI System...")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model - using your existing approach
    print("Loading ResNet18 model...")
    model = models.resnet18(weights=None)  # Updated from pretrained=False
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
    
    # Setup GradCAM - using your existing GradCAM class
    target_layer = model.layer4[1].conv2
    gradcam = GradCAM(model, target_layer)
    print("GradCAM initialized")
    
    # Setup transforms - matching your existing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Initialize QA chain - using your existing setup
    try:
        qa_chain = get_qa_chain(temperature=0.0)
        print("LLM QA chain initialized")
    except Exception as e:
        print(f"Could not initialize QA chain: {e}")
        qa_chain = None
    
    print("System startup complete!")

def analyze_gradcam_regions(heatmap, pred_class):
    """Analyze GradCAM heatmap to identify key regions - from your existing code"""
    height, width = heatmap.shape
    max_activation = np.max(heatmap)
    threshold = max_activation * 0.7  # Focus on top 30% of activations
    
    # Find activated regions
    activated_mask = heatmap > threshold
    
    # Determine anatomical regions based on heatmap location
    regions = []
    
    # Divide image into anatomical regions
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
    
    # Generate description based on prediction and regions
    if pred_class == 1:  # PNEUMONIA
        if regions:
            region_text = ", ".join(regions)
            description = f"Chest X-ray shows abnormal opacities in the {region_text}. AI model detected features consistent with pneumonia with high confidence. The areas of concern show increased density patterns typical of inflammatory processes."
        else:
            description = "Chest X-ray shows features suggestive of pneumonia. AI analysis indicates abnormal lung parenchyma consistent with infectious process."
    else:  # NORMAL
        description = "Chest X-ray appears within normal limits. Lung fields are clear with no evidence of acute cardiopulmonary abnormalities. AI analysis supports normal chest radiograph findings."
    
    return description, regions

def generate_comprehensive_report(image_analysis, pred_class, confidence, regions):
    """Generate detailed medical report sections - from your existing code"""
    if qa_chain is None:
        # Fallback if LLM is not available
        if pred_class == 1:
            findings = "Abnormal chest radiograph findings consistent with pneumonic consolidation."
            impression = "Findings consistent with pneumonia."
            recommendations = "Clinical correlation recommended. Consider antibiotic therapy if clinically indicated."
        else:
            findings = "Normal chest radiograph. Clear lung fields bilaterally."
            impression = "No acute cardiopulmonary abnormalities."
            recommendations = "Routine follow-up as clinically indicated."
        return findings, impression, recommendations
    
    # Use LLM for detailed reports - matching your existing prompts
    findings_prompt = f"""
    Based on this chest X-ray analysis: {image_analysis}
    
    Generate detailed radiological findings in proper medical terminology. 
    Include specific observations about lung fields, heart size, and any abnormalities noted.
    Keep it concise and professional.
    """
    findings = ask_qa(findings_prompt, qa_chain)
    
    # Generate impression section
    if pred_class == 1:  # PNEUMONIA
        impression_prompt = f"""
        Given these chest X-ray findings showing pneumonia in regions: {', '.join(regions) if regions else 'multiple areas'}
        
        Provide a clinical impression statement that would be appropriate for a radiologist's report.
        Include the likely diagnosis and any relevant clinical correlations needed.
        """
    else:  # NORMAL
        impression_prompt = """
        For a normal chest X-ray with clear lung fields and no abnormalities,
        provide a standard radiological impression statement.
        """
    
    impression = ask_qa(impression_prompt, qa_chain)
    
    # Generate recommendations
    if pred_class == 1:  # PNEUMONIA
        recommendations_prompt = """
        For a chest X-ray showing findings consistent with pneumonia,
        what clinical recommendations should a radiologist include?
        Focus on follow-up imaging, clinical correlation, and next steps.
        """
        recommendations = ask_qa(recommendations_prompt, qa_chain)
    else:  # NORMAL
        recommendations = "No acute findings. Routine clinical correlation as indicated."
    
    return findings, impression, recommendations

def get_additional_info(condition):
    """Generate condition-specific information - from your existing code"""
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
#can run other functions while waiting for some slow operations inside this function
 
async def process_xray_analysis(analysis_id: str):
    """Background task to process X-ray analysis - adapted from your existing main_pipeline.py"""
    try:
        # Update status
        analysis_store[analysis_id]["status"] = AnalysisStatus.PROCESSING
        analysis_store[analysis_id]["updated_at"] = datetime.now().isoformat()
        
        # Get file path
        file_path = analysis_store[analysis_id]["file_path"]
        filename = analysis_store[analysis_id]["filename"]
        
        # Load and process image - matching your existing pipeline
        img = Image.open(file_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # Forward pass
        output = model(input_tensor)
        pred_class = torch.argmax(output, 1).item()
        confidence = torch.softmax(output, 1)[0, pred_class].item()
        
        # Backward pass for GradCAM
        score = output[0, pred_class]
        model.zero_grad()
        score.backward()
        
        # Generate Grad-CAM heatmap - using your existing method
        heatmap = gradcam.generate_heatmap(pred_class)
        heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        img_np = np.array(img)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)
        
        # Save GradCAM overlay - matching your existing file structure
        gradcam_path = GRADCAM_DIR / f"API_{analysis_id}_{filename}_gradcam.jpg"
        cv2.imwrite(str(gradcam_path), overlay)
        
        # Analyze regions and generate report - using your existing functions
        image_analysis, regions = analyze_gradcam_regions(heatmap, pred_class)
        findings, impression, recommendations = generate_comprehensive_report(
            image_analysis, pred_class, confidence, regions
        )
        causes, treatments, prevention = get_additional_info(classes[pred_class])
        
        # Create prediction summary
        prediction_summary = f"""Model Classification: {classes[pred_class]}
Confidence Score: {confidence:.3f} ({confidence*100:.1f}%)
Analysis Method: ResNet18 CNN with GradCAM visualization
Affected Regions: {', '.join(regions) if regions else 'Diffuse lung fields'}
Image: {filename}"""
        
        # Create full report - matching your existing format
        full_report = f"""Findings:
{findings}
- Highlighted regions on Grad-CAM show attention focused on: {', '.join(regions) if regions else 'diffuse lung fields'}
- AI model prediction: {classes[pred_class]} (confidence: {confidence:.2f})

Impression:
{impression}

Recommendations:
{recommendations}"""
        
        # Structure final report - using your existing PneumoniaInfo schema
        report_obj = PneumoniaInfo(
            prediction=prediction_summary.strip(),
            findings=full_report.strip(),
            causes=causes,
            treatments=treatments,
            prevention=prevention
        )
        
        # Save report JSON - matching your existing file structure
        report_path = REPORT_DIR / f"API_{analysis_id}_{filename}_report.json"
        with open(report_path, "w") as f:
            f.write(report_obj.model_dump_json(indent=2))
        
        # Update analysis record
        analysis_store[analysis_id].update({
            "status": AnalysisStatus.COMPLETED,
            "updated_at": datetime.now().isoformat(),
            "prediction": classes[pred_class],
            "confidence": confidence,
            "regions": regions,
            "report_path": str(report_path),
            "gradcam_path": str(gradcam_path),
            "report_data": report_obj.model_dump(),
            "processing_time": (datetime.now() - datetime.fromisoformat(analysis_store[analysis_id]["created_at"])).total_seconds()
        })
        
        print(f"Analysis {analysis_id} completed: {classes[pred_class]} ({confidence:.2f})")
        
    except Exception as e:
        print(f"Analysis {analysis_id} failed: {str(e)}")
        analysis_store[analysis_id].update({
            "status": AnalysisStatus.FAILED,
            "updated_at": datetime.now().isoformat(),
            "error": str(e)
        })

@app.post("/api/analyze-xray", response_model=AnalysisResponse)
async def analyze_xray(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: Optional[str] = None,
    patient_name: Optional[str] = None
):
    """Upload X-ray image for analysis"""
    
    # Validate file
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Generate analysis ID
    analysis_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{analysis_id}_{file.filename}"
    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)
    
    # Create analysis record
    analysis_record = {
        "id": analysis_id,
        "filename": file.filename,
        "file_path": str(file_path),
        "patient_id": patient_id,
        "patient_name": patient_name,
        "status": AnalysisStatus.PENDING,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    analysis_store[analysis_id] = analysis_record
    
    # Queue processing
    background_tasks.add_task(process_xray_analysis, analysis_id)
    
    return AnalysisResponse(
        analysis_id=analysis_id,
        status=AnalysisStatus.PENDING,
        message="Analysis queued for processing",
        created_at=analysis_record["created_at"]
    )

@app.get("/api/analysis/{analysis_id}")
async def get_analysis_status(analysis_id: str):
    """Get analysis status and results"""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analysis_store[analysis_id]
    
    # Include report data if completed
    response = {
        "id": analysis_id,
        "status": analysis["status"],
        "created_at": analysis["created_at"],
        "updated_at": analysis["updated_at"],
        "filename": analysis["filename"],
        "patient_info": {
            "id": analysis.get("patient_id"),
            "name": analysis.get("patient_name")
        }
    }
    
    if analysis["status"] == AnalysisStatus.COMPLETED:
        response.update({
            "prediction": analysis.get("prediction"),
            "confidence": analysis.get("confidence"),
            "regions": analysis.get("regions"),
            "processing_time": analysis.get("processing_time"),
            "gradcam_url": f"/api/analysis/{analysis_id}/gradcam",
            "report_url": f"/api/analysis/{analysis_id}/report"
        })
    elif analysis["status"] == AnalysisStatus.FAILED:
        response["error"] = analysis.get("error")
    
    return response

@app.get("/api/analysis/{analysis_id}/report")
async def get_analysis_report(analysis_id: str):
    """Get detailed analysis report"""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analysis_store[analysis_id]
    
    if analysis["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Analysis not completed. Status: {analysis['status']}")
    
    # Return the structured report data
    return analysis["report_data"]

@app.get("/api/analysis/{analysis_id}/gradcam")
async def get_gradcam_image(analysis_id: str):
    """Get GradCAM visualization"""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analysis_store[analysis_id]
    
    if analysis["status"] != AnalysisStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Analysis not completed")
    
    gradcam_path = analysis.get("gradcam_path")
    if gradcam_path and os.path.exists(gradcam_path):
        return FileResponse(gradcam_path, media_type="image/jpeg")
    
    raise HTTPException(status_code=404, detail="GradCAM image not found")

@app.get("/api/analyses")
async def list_analyses(limit: int = 10, offset: int = 0):
    """List all analyses with pagination"""
    analyses = list(analysis_store.values())
    analyses.sort(key=lambda x: x["created_at"], reverse=True)
    
    total = len(analyses)
    page_analyses = analyses[offset:offset + limit]
    
    # Return summary data for list view
    return {
        "total": total,
        "limit": limit, 
        "offset": offset,
        "analyses": [
            {
                "id": a["id"],
                "filename": a["filename"],
                "status": a["status"],
                "created_at": a["created_at"],
                "patient_name": a.get("patient_name"),
                "prediction": a.get("prediction"),
                "confidence": a.get("confidence")
            } for a in page_analyses
        ]
    }

@app.delete("/api/analysis/{analysis_id}")
async def delete_analysis(analysis_id: str):
    """Delete analysis and files"""
    if analysis_id not in analysis_store:
        raise HTTPException(status_code=404, detail="Analysis not found")
    
    analysis = analysis_store[analysis_id]
    
    # Delete files
    try:
        # Delete uploaded file
        if os.path.exists(analysis["file_path"]):
            os.remove(analysis["file_path"])
        
        # Delete gradcam and report if they exist
        gradcam_path = analysis.get("gradcam_path")
        if gradcam_path and os.path.exists(gradcam_path):
            os.remove(gradcam_path)
            
        report_path = analysis.get("report_path")
        if report_path and os.path.exists(report_path):
            os.remove(report_path)
            
    except Exception as e:
        print(f"Error deleting files: {e}")
    
    del analysis_store[analysis_id]
    return {"message": "Analysis deleted successfully"}

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "gpu_available": torch.cuda.is_available(),
        "model_loaded": model is not None,
        "qa_chain_available": qa_chain is not None,
        "total_analyses": len(analysis_store)
    }

@app.get("/")
async def root():
    """Root endpoint with API information"""
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
        reload=False,  # Set to True for development
        workers=1
    )
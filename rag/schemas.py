# rag/schemas.py

from pydantic import BaseModel

class PneumoniaInfo(BaseModel):
    prediction: str
    findings: str
    causes: str
    treatments: str
    prevention: str

class RadiologyReport(BaseModel):
    patient_id: str
    findings: str
    impression: str
    recommendation: str

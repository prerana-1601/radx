import streamlit as st
import requests
from PIL import Image
import io

# Set API base URL
#API_URL = "http://localhost:8000/api"

import os
API_URL = os.getenv("API_URL", "http://localhost:8000/api")

st.title("RadX-CV Medical AI System")
st.write("Upload a chest X-ray image to get AI-assisted analysis and report.")

# Patient metadata inputs
st.subheader("Patient Metadata (optional)")
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", min_value=0, max_value=120, value=50, step=1)
    sex = st.selectbox("Sex", ["Male", "Female", "Other"], index=0)
with col2:
    smoker = st.selectbox("Smoking status", ["Non-smoker", "Former smoker", "Current smoker"], index=0)

symptoms = st.multiselect(
    "Presenting symptoms",
    ["cough", "fever", "chest pain", "breathlessness", "hemoptysis"],
    default=["cough", "fever"]
)

col3, col4 = st.columns(2)
with col3:
    duration = st.selectbox("Duration of symptoms", ["acute (<2 weeks)", "chronic (>2 weeks)"])
with col4:
    severity = st.selectbox("Severity", ["mild", "moderate", "severe"], index=0)

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded X-ray", use_container_width=True)
    
    # Send file to API
    if st.button("Analyze X-ray"):
        with st.spinner("Uploading and processing..."):
            files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
            data = {
                "patient_age": str(age),
                "patient_sex": sex,
                "smoking_status": smoker,
                "symptoms": ",".join(symptoms),
                "symptom_duration": duration,
                "symptom_severity": severity,
            }
            response = requests.post(f"{API_URL}/analyze-xray", files=files, data=data)
            
            if response.status_code == 200:
                analysis_id = response.json()["analysis_id"]
                st.success(f"Analysis queued! ID: {analysis_id}")
                
                # Polling for completion
                import time
                status = "pending"
                while status in ["pending", "processing"]:
                    time.sleep(2)  # wait 2 seconds before polling
                    status_resp = requests.get(f"{API_URL}/analysis/{analysis_id}")
                    status = status_resp.json()["status"]
                
                if status == "completed":
                    st.success("Analysis completed!")
                    
                    # GradCAM image
                    gradcam_resp = requests.get(f"{API_URL}/analysis/{analysis_id}/gradcam")
                    gradcam_image = Image.open(io.BytesIO(gradcam_resp.content))
                    st.image(gradcam_image, caption="GradCAM Overlay", use_container_width=True)
                    
                    # Report
                    report_resp = requests.get(f"{API_URL}/analysis/{analysis_id}/report")
                    report = report_resp.json()
                    
                    st.subheader("Prediction Summary")
                    st.text(report["prediction"])
                    
                    st.subheader("Detailed Report")
                    st.text(report["findings"])
                    
                    st.subheader("Causes, Treatments, Prevention")
                    st.markdown(f"**Causes:** {report['causes']}")
                    st.markdown(f"**Treatments:** {report['treatments']}")
                    st.markdown(f"**Prevention:** {report['prevention']}")
                
                else:
                    st.error(f"Analysis failed. Status: {status}")
            else:
                st.error(f"Failed to submit image: {response.text}")

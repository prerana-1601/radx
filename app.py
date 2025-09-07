import streamlit as st
import requests
from PIL import Image
import io

# Set API base URL
API_URL = "http://localhost:8000/api"

st.title("RadX-CV Medical AI System")
st.write("Upload a chest X-ray image to get AI-assisted analysis and report.")

# Upload image
uploaded_file = st.file_uploader("Choose an X-ray image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded X-ray", use_column_width=True)
    
    # Send file to API
    if st.button("Analyze X-ray"):
        with st.spinner("Uploading and processing..."):
            files = {"file": (uploaded_file.name, uploaded_file, "image/jpeg")}
            response = requests.post(f"{API_URL}/analyze-xray", files=files)
            
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

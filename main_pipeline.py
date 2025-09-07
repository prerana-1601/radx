import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
from rag.llm_chain import get_qa_chain, _extract_text, ask_qa
from rag.schemas import PneumoniaInfo
from rag.llm_chain import generate_radiology_report
from api.gradcam import GradCAM

# ------------------
# Setup
# ------------------
print("Setting up model and GradCAM...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("models/resnet18_xray.pth", map_location=device))
model = model.to(device)
model.eval()

# The layer from which the feature map is extracted
target_layer = model.layer4[1].conv2

# Creating a GradCAM object
gradcam = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------
# Paths
# ------------------
data_dir = "data/chest_xray/test"
gradcam_dir = "outputs/gradcam"
report_dir = "outputs/reports"
os.makedirs(gradcam_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

classes = ["NORMAL", "PNEUMONIA"]
qa_chain = get_qa_chain(temperature=0.0)

def analyze_gradcam_regions(heatmap, pred_class):
    """
    Analyze the GradCAM heatmap to identify key regions and generate descriptive text
    """
    # Find areas of highest activation
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

def generate_comprehensive_report(image_analysis, pred_class, confidence_score, regions):
    """
    Generate different sections of the report using the QA chain
    """
    # Generate findings section
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
    """
    Generate condition-specific information for causes, treatments, and prevention
    """
    if condition == "PNEUMONIA":
        causes_prompt = "What are the main causes of pneumonia? Provide a concise medical overview."
        treatments_prompt = "What are the standard treatments for pneumonia? Include both outpatient and inpatient approaches."
        prevention_prompt = "How can pneumonia be prevented? List key prevention strategies."
        
        causes = ask_qa(causes_prompt, qa_chain)
        treatments = ask_qa(treatments_prompt, qa_chain)
        prevention = ask_qa(prevention_prompt, qa_chain)
    else:  # NORMAL
        causes = "No pathological condition detected."
        treatments = "No treatment required for normal chest X-ray findings."
        prevention = "Maintain good respiratory health through regular exercise, vaccination, and avoiding smoking."
    
    return causes, treatments, prevention

# ------------------
# Main Loop
# ------------------
for label in classes:
    print(f"Processing {label} images...")
    folder = os.path.join(data_dir, label)
    
    for img_name in os.listdir(folder):
        print(f"Processing {img_name}...")
        img_path = os.path.join(folder, img_name)
        img = Image.open(img_path).convert("RGB")
        input_tensor = transform(img).unsqueeze(0).to(device)

        # Forward pass
        output = model(input_tensor)
        pred_class = torch.argmax(output, 1).item()
        confidence = torch.softmax(output, 1)[0, pred_class].item()
        
        # Backward pass for GradCAM
        score = output[0, pred_class]
        model.zero_grad()
        score.backward()

        # Generate Grad-CAM heatmap
        heatmap = gradcam.generate_heatmap(pred_class)
        heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
        heatmap_uint8 = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        img_np = np.array(img)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap_colored, 0.4, 0)

        # Save overlay
        gradcam_path = os.path.join(gradcam_dir, f"{label}_{img_name}_gradcam.jpg")
        cv2.imwrite(gradcam_path, overlay)

        # Analyze GradCAM regions and generate description
        image_analysis, regions = analyze_gradcam_regions(heatmap, pred_class)
        
        # Generate comprehensive report sections
        findings, impression, recommendations = generate_comprehensive_report(
            image_analysis, pred_class, confidence, regions
        )
        
        # Get additional medical information
        causes, treatments, prevention = get_additional_info(classes[pred_class])
        
        # Create structured report with dynamic content
        full_report = f"""
Findings:
{findings}
- Highlighted regions on Grad-CAM show attention focused on: {', '.join(regions) if regions else 'diffuse lung fields'}
- AI model prediction: {classes[pred_class]} (confidence: {confidence:.2f})

Impression:
{impression}

Recommendations:
{recommendations}
"""
        # Create prediction summary
        prediction_summary = f"""
Model Classification: {classes[pred_class]}
Confidence Score: {confidence:.3f} ({confidence*100:.1f}%)
Analysis Method: ResNet18 CNN with GradCAM visualization
Image: {img_name}
"""
        
        # Structure report with all dynamic content
        report_obj = PneumoniaInfo(
            prediction=prediction_summary.strip(),
            findings=full_report.strip(),
            causes=causes,
            treatments=treatments,
            prevention=prevention
        )
        
        # Save report as JSON
        report_path = os.path.join(report_dir, f"{label}_{img_name}_report.json")
        with open(report_path, "w") as f:
            f.write(report_obj.model_dump_json(indent=2))

        print(f"✅ Processed {img_name}")
        print(f"   Prediction: {classes[pred_class]} (confidence: {confidence:.2f})")
        print(f"   Regions: {', '.join(regions) if regions else 'General lung fields'}")
        print(f"   GradCAM saved: {gradcam_path}")
        print(f"   Report saved: {report_path}")
        print("-" * 50)

print("Pipeline completed!")
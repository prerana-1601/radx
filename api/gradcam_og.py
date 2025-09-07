'''
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Function
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

# ------------------
# Grad-CAM Hook
# ------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)

    def generate(self, class_idx):
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]

        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

# ------------------
# Setup
# ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet18(pretrained=False)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 2)
model.load_state_dict(torch.load("models/resnet18_xray.pth", map_location=device))
model = model.to(device)
model.eval()

# Attach Grad-CAM to last conv layer
target_layer = model.layer4[1].conv2
gradcam = GradCAM(model, target_layer)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ------------------
# Run Grad-CAM on test images
# ------------------
data_dir = "data/chest_xray/test"
output_dir = "outputs/gradcam"
os.makedirs(output_dir, exist_ok=True)

classes = ["NORMAL", "PNEUMONIA"]

for label in classes:
    folder = os.path.join(data_dir, label)
    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)
        img = Image.open(img_path).convert("RGB")

        input_tensor = transform(img).unsqueeze(0).to(device)

        # Forward + Backward
        output = model(input_tensor)
        pred_class = torch.argmax(output, 1).item()
        score = output[0, pred_class]

        model.zero_grad()
        score.backward()

        # Generate heatmap
        heatmap = gradcam.generate(pred_class)

        # Resize heatmap to image size
        heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        img_np = np.array(img)
        overlay = cv2.addWeighted(img_np, 0.6, heatmap, 0.4, 0)

        # Save
        save_path = os.path.join(output_dir, f"{label}_{img_name}_gradcam.jpg")
        cv2.imwrite(save_path, overlay)

        print(f"Processed {img_name} -> Pred: {classes[pred_class]}")
'''
# api/gradcam.py

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

# ------------------
# Grad-CAM Hook
# ------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        #under the hood 
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        #register_forward_hook pytorch method that calls the forward_hook every time target_layer is run
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, class_idx):
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        #print("hi")
        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        return heatmap

    def generate_report(self, pred_label: str, heatmap: np.ndarray) -> str:
        """
        Convert Grad-CAM heatmap + prediction into a textual summary
        """
        # Find high-activation region (simplified)
        region_focus = "upper lung region" if heatmap[:heatmap.shape[0]//2, :].mean() > heatmap[heatmap.shape[0]//2:, :].mean() else "lower lung region"
        report = (
            f"The chest X-ray shows features consistent with {pred_label.lower()}. "
            f"Highlighted regions on Grad-CAM suggest attention is focused on the {region_focus}. "
            f"Clinical correlation is recommended."
        )
        print(report)
        return report


# ------------------
# Model Setup
# ------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    model.load_state_dict(torch.load("models/resnet18_xray.pth", map_location=device))
    model = model.to(device)
    model.eval()

    target_layer = model.layer4[1].conv2
    gradcam = GradCAM(model, target_layer)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    # ------------------
    # Inference & report generation
    # ------------------
    data_dir = "data/chest_xray/test"
    output_dir = "outputs/gradcam"
    os.makedirs(output_dir, exist_ok=True)

    classes = ["NORMAL", "PNEUMONIA"]

    reports = []  # store textual summaries

    for label in classes:
        folder = os.path.join(data_dir, label)
        for img_name in os.listdir(folder):
            img_path = os.path.join(folder, img_name)
            img = Image.open(img_path).convert("RGB")

            input_tensor = transform(img).unsqueeze(0).to(device)

            # Forward + Backward
            output = model(input_tensor)
            pred_class = torch.argmax(output, 1).item()
            score = output[0, pred_class]

            model.zero_grad()
            score.backward()

            heatmap = gradcam.generate_heatmap(pred_class)
            report_text = gradcam.generate_report(classes[pred_class], heatmap)
            reports.append((img_name, report_text))

            # Save overlay
            heatmap_resized = cv2.resize(heatmap, (img.size[0], img.size[1]))
            heatmap_resized = np.uint8(255 * heatmap_resized)
            heatmap_resized = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(np.array(img), 0.6, heatmap_resized, 0.4, 0)
            cv2.imwrite(os.path.join(output_dir, f"{label}_{img_name}_gradcam.jpg"), overlay)

            print(f"Processed {img_name} -> Pred: {classes[pred_class]}")

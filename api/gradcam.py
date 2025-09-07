# api/gradcam.py - Updated for better compatibility

import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

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

        # Use register_full_backward_hook for better compatibility
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_heatmap(self, class_idx):
        """Generate GradCAM heatmap for the given class index"""
        if self.gradients is None or self.activations is None:
            raise ValueError("No gradients or activations found. Make sure to run backward pass first.")
        
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        activations = self.activations[0]
        
        for i in range(len(pooled_gradients)):
            activations[i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(activations, dim=0).cpu().numpy()
        heatmap = np.maximum(heatmap, 0)
        
        # Avoid division by zero
        if np.max(heatmap) > 0:
            heatmap /= np.max(heatmap)
        
        return heatmap

    def generate_report(self, pred_label: str, heatmap: np.ndarray) -> str:
        """
        Convert Grad-CAM heatmap + prediction into a textual summary
        """
        # Determine focus region based on heatmap
        height, width = heatmap.shape
        upper_half_activation = heatmap[:height//2, :].mean()
        lower_half_activation = heatmap[height//2:, :].mean()
        
        if upper_half_activation > lower_half_activation:
            region_focus = "upper lung region"
        else:
            region_focus = "lower lung region"
        
        # More detailed region analysis
        left_half_activation = heatmap[:, :width//2].mean()
        right_half_activation = heatmap[:, width//2:].mean()
        
        if abs(left_half_activation - right_half_activation) > 0.1:
            if left_half_activation > right_half_activation:
                region_focus = f"left {region_focus}"
            else:
                region_focus = f"right {region_focus}"
        
        report = (
            f"The chest X-ray shows features consistent with {pred_label.lower()}. "
            f"Highlighted regions on Grad-CAM suggest attention is focused on the {region_focus}. "
            f"Clinical correlation is recommended."
        )
        
        return report

    # Backward compatibility - keep the old method name
    def generate(self, class_idx):
        """Backward compatibility with old method name"""
        return self.generate_heatmap(class_idx)


# Standalone execution for testing (matches your existing pattern)
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=None)  # Updated from pretrained=False
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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Test processing
    data_dir = "data/chest_xray/test"
    output_dir = "outputs/gradcam"
    os.makedirs(output_dir, exist_ok=True)

    classes = ["NORMAL", "PNEUMONIA"]
    reports = []

    for label in classes:
        folder = os.path.join(data_dir, label)
        if not os.path.exists(folder):
            print(f"Folder {folder} does not exist, skipping...")
            continue
            
        for img_name in os.listdir(folder):
            try:
                img_path = os.path.join(folder, img_name)
                img = Image.open(img_path).convert("RGB")
                input_tensor = transform(img).unsqueeze(0).to(device)

                # Forward + Backward
                output = model(input_tensor)
                pred_class = torch.argmax(output, 1).item()
                score = output[0, pred_class]

                model.zero_grad()
                score.backward()

                # Generate heatmap and report
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
                
            except Exception as e:
                print(f"Error processing {img_name}: {e}")
                continue

    print(f"\nGenerated {len(reports)} reports")
    for img_name, report in reports[:3]:  # Show first 3 reports
        print(f"\n{img_name}: {report}")
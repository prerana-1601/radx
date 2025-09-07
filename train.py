# train.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# --------------------------
# Step 1: Set device
# --------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --------------------------
# Step 2: Define paths
# --------------------------
base_dir = os.path.dirname(os.path.abspath(__file__)) 
data_dir = os.path.join(base_dir,"data/chest_xray")
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")
model_save_path = os.path.join(base_dir, "models/resnet18_xray.pth")

# --------------------------
# Step 3: Define transforms
# --------------------------
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),    # Resize images to 224x224 for ResNet
    transforms.RandomHorizontalFlip(), # Data augmentation
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]) # Imagenet normalization
])

val_test_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# --------------------------
# Step 4: Load datasets
# --------------------------
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
val_dataset = datasets.ImageFolder(val_dir, transform=val_test_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=val_test_transforms)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False,  num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False,  num_workers=2)

# --------------------------
# Step 5: Define model
# --------------------------
model = models.resnet18(pretrained=True)       # Use pretrained ResNet18
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)         # 2 classes: NORMAL, PNEUMONIA
model = model.to(device)

# --------------------------
# Step 6: Loss and optimizer
# --------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --------------------------
# Step 7: Training loop
# --------------------------
num_epochs = 5
best_val_acc = 0

for epoch in range(num_epochs):
    model.train()
    train_losses = []
    all_preds = []
    all_labels = []
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        train_losses.append(loss.item())
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    train_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {np.mean(train_losses):.4f}, Train Acc: {train_acc:.4f}")
    
    # Validation
    model.eval()
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())
    val_acc = accuracy_score(val_labels, val_preds)
    print(f"Validation Accuracy: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), model_save_path)
        print(f"Saved best model with val acc: {best_val_acc:.4f}")

# --------------------------
# Step 8: Test evaluation
# --------------------------
model.load_state_dict(torch.load(model_save_path))
model.eval()
test_preds = []
test_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_labels.extend(labels.cpu().numpy())

print("Test Classification Report:")
print(classification_report(test_labels, test_preds, target_names=train_dataset.classes))

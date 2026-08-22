import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import BloodMNIST
import pandas as pd
import random
import numpy as np

# --- 0. Set Seeds for Reproducibility ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- 1. Hyperparameters ---
EPOCHS = 50
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Running Ultimate Centralized Baseline on: {DEVICE}")

# --- 2. Data Preparation (Fair Apples-to-Apples Augmentation) ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

os.makedirs("./data", exist_ok=True)
train_dataset = BloodMNIST(split="train", transform=train_transform, download=True, root="./data")
test_dataset = BloodMNIST(split="test", transform=test_transform, download=True, root="./data")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Model Architecture ---
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False
for param in model.layer4.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 8)
model = model.to(DEVICE)

# --- 4. Training Loop with BEST Checkpoint Saving ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

best_acc = 0.0
history = {"Round": [], "Loss": [], "Accuracy": []} # Header named 'Round' to match Federated CSVs easily

for epoch in range(EPOCHS):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    # Evaluation
    model.eval()
    val_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
            outputs = model(images)
            val_loss += criterion(outputs, labels).item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_acc = correct / total
    epoch_loss = val_loss / total
    
    history["Round"].append(epoch + 1)
    history["Loss"].append(epoch_loss)
    history["Accuracy"].append(epoch_acc)

    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc*100:.2f}%")

    if epoch_acc > best_acc:
        print(f"  🌟 New Best Accuracy: {epoch_acc*100:.2f}% (Previous: {best_acc*100:.2f}%). Saving model...")
        best_acc = epoch_acc
        torch.save(model.state_dict(), "Ultimate_Centralized_Baseline.pth")

# --- 5. Save Logs ---
df = pd.DataFrame(history)
df.to_csv("Ultimate_Centralized_history.csv", index=False)
print(f"\n✅ Training complete! The absolute best model achieved {best_acc*100:.2f}% and is saved as 'Ultimate_Centralized_Baseline.pth'")
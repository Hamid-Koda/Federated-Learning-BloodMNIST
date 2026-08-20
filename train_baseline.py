import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import BloodMNIST
import matplotlib.pyplot as plt
import os

# 1. Hyperparameters & Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  
EPOCHS = 50      
LR = 0.001

# 2. Data Pre-processing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Downloading Datasets...")
os.makedirs("./data", exist_ok=True)
train_dataset = BloodMNIST(split='train', transform=train_transform, download=True, root='./data')
test_dataset = BloodMNIST(split='test', transform=test_transform, download=True, root='./data')

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Model Architecture 
print("Configuring ResNet18 (Unfreezing Layer 4 & FC)...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

for param in model.layer4.parameters():
    param.requires_grad = True

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8) 
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# 4. Training Loop
history_loss, history_acc = [], []
print(f"Centralized Training started on {DEVICE}...")

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    epoch_loss = total_loss / len(train_loader)
    history_loss.append(epoch_loss)
    
    # Evaluation
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_acc = correct / total
    history_acc.append(epoch_acc)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {epoch_loss:.4f} - Test Accuracy: {epoch_acc*100:.2f}%")

# 5. Save the baseline model weights
torch.save(model.state_dict(), 'resnet18_centralized.pth')
print("\n✓ Baseline model saved as 'resnet18_centralized.pth'")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import BloodMNIST
import matplotlib.pyplot as plt

# 1. Hyperparameters & Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  
EPOCHS = 5
LR = 0.001

# 2. Data Pre-processing (Upgraded for ResNet18)
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

train_dataset = BloodMNIST(split='train', transform=train_transform, download=True, root='./data')
test_dataset = BloodMNIST(split='test', transform=test_transform, download=True, root='./data')

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Transfer Learning: ResNet18 Architecture
print("Downloading and configuring pre-trained ResNet18...")
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

for param in model.parameters():
    param.requires_grad = False

num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8) 
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

history_loss = []
history_acc = []

# 4. Training Loop
print(f"ResNet18 training started on {DEVICE}...")
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
    correct = 0
    total = 0
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

# 5. Save the ResNet18 model weights
model_filename = 'resnet18_blood_model.pth'
torch.save(model.state_dict(), model_filename)
print(f"\n✓ ResNet18 weights successfully saved as '{model_filename}'")

# 6. Plotting
print("Generating ResNet18 Metrics Plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(range(1, EPOCHS + 1), history_loss, marker='o', color='darkred', linewidth=2)
ax1.set_title('ResNet18 Training Loss')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--')

ax2.plot(range(1, EPOCHS + 1), [a * 100 for a in history_acc], marker='s', color='forestgreen', linewidth=2)
ax2.set_title('ResNet18 Testing Accuracy')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, linestyle='--')

plt.tight_layout()
plt.savefig("centralized_resnet18_plot.png", dpi=300)
print("✓ Plot saved as 'centralized_resnet18_plot.png'")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from medmnist import BloodMNIST
import matplotlib.pyplot as plt

# 1. Hyperparameters & Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
EPOCHS = 5
LR = 0.001

# 2. Data Pre-processing & Transformations
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load train and test splits of BloodMNIST
train_dataset = BloodMNIST(split='train', transform=data_transform, download=True, root='./data')
test_dataset = BloodMNIST(split='test', transform=data_transform, download=True, root='./data')

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. Centralized CNN Model Architecture
class BloodCNN(nn.Module):
    def __init__(self):
        super(BloodCNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

model = BloodCNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# Lists to log metrics history for plotting
history_loss = []
history_acc = []

# 4. Training & Evaluation Loop
print(f"Centralized training started on {DEVICE}...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    epoch_loss = total_loss / len(train_loader)
    history_loss.append(epoch_loss)
    
    # Evaluate on test set after each epoch to track accuracy progress
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

# 5. Plotting and Saving Centralized Metrics
print("\nGenerating Centralized Training Plots...")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot training loss history
ax1.plot(range(1, EPOCHS + 1), history_loss, marker='o', color='darkblue', linewidth=2, label='Train Loss')
ax1.set_title('Centralized Model Loss over Epochs', fontsize=12, fontweight='bold')
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

# Plot testing accuracy history
ax2.plot(range(1, EPOCHS + 1), [a * 100 for a in history_acc], marker='s', color='darkorange', linewidth=2, label='Test Accuracy')
ax2.set_title('Centralized Model Accuracy over Epochs', fontsize=12, fontweight='bold')
ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()

# Save the plot image locally
plot_filename = "centralized_metrics_plot.png"
plt.savefig(plot_filename, dpi=300)
print(f"✓ Success! Centralized plot saved automatically as '{plot_filename}'")
plt.show()
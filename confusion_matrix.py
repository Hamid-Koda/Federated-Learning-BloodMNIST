import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import BloodMNIST
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import os

# --- 1. Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64

# نام کلاس‌های دیتاست BloodMNIST به ترتیب
CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 
           'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

# --- 2. Data Preparation (Only Test Set Needed) ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading Test Dataset...")
test_dataset = BloodMNIST(split="test", transform=test_transform, download=True, root="./data")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Model Architecture (Must match training exactly) ---
def get_resnet18():
    model = models.resnet18(weights=None) # نیازی به دانلود مجدد وزن‌های ایمیج‌نت نیست
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8) 
    return model

# --- 4. Evaluation Function ---
def evaluate_model(model_path, model_name):
    if not os.path.exists(model_path):
        print(f"❌ Error: Model file '{model_path}' not found!")
        return None, None
        
    model = get_resnet18().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    print(f"Evaluating {model_name}...")
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    # چاپ گزارش متنی کامل (Precision, Recall, F1)
    print(f"\n--- Classification Report for {model_name} ---")
    print(classification_report(all_labels, all_preds, target_names=CLASSES))
    
    return all_labels, all_preds

# --- 5. Run Evaluations ---
labels_fedavg, preds_fedavg = evaluate_model("fedavg_global_model.pth", "FedAvg")
labels_fedprox, preds_fedprox = evaluate_model("fedprox_global_model.pth", "FedProx")

# --- 6. Plotting Confusion Matrices ---
if labels_fedavg and labels_fedprox:
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # FedAvg Matrix
    cm_fedavg = confusion_matrix(labels_fedavg, preds_fedavg)
    sns.heatmap(cm_fedavg, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=CLASSES, yticklabels=CLASSES)
    axes[0].set_title('FedAvg Global Model Confusion Matrix', fontweight='bold')
    axes[0].set_ylabel('True Label')
    axes[0].set_xlabel('Predicted Label')
    axes[0].tick_params(axis='x', rotation=45)
    
    # FedProx Matrix
    cm_fedprox = confusion_matrix(labels_fedprox, preds_fedprox)
    sns.heatmap(cm_fedprox, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                xticklabels=CLASSES, yticklabels=CLASSES)
    axes[1].set_title('FedProx Global Model Confusion Matrix', fontweight='bold')
    axes[1].set_ylabel('True Label')
    axes[1].set_xlabel('Predicted Label')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig("federated_confusion_matrices.png", dpi=300, bbox_inches='tight')
    print("\n✓ Plot saved successfully as 'federated_confusion_matrices.png'")
    plt.show()
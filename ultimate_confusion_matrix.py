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
CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 
           'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

# --- 2. Data Preparation ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading Test Dataset...")
test_dataset = BloodMNIST(split="test", transform=test_transform, download=True, root="./data")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Model Architecture ---
def get_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8) 
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
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    report = classification_report(all_labels, all_preds, target_names=CLASSES)
    with open(f"{model_name.replace(' ', '_')}_Report.txt", "w") as f:
        f.write(f"--- Classification Report for {model_name} ---\n\n")
        f.write(report)
    print(f"✓ {model_name} evaluated and report saved.")
    
    return all_labels, all_preds

# --- 5. Run Evaluations ---
labels_base, preds_base = evaluate_model("resnet18_centralized.pth", "Centralized Baseline")
labels_avg, preds_avg = evaluate_model("fedavg_global_model.pth", "FedAvg Global")
labels_prox, preds_prox = evaluate_model("fedprox_global_model.pth", "FedProx Global")

# --- 6. Plotting Normalized Confusion Matrices ---
if labels_base is not None and labels_avg is not None and labels_prox is not None:
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    
    def plot_normalized_cm(labels, preds, ax, title, cmap):
        cm = confusion_matrix(labels, preds, normalize='true') 
        cm_percentage = cm * 100 
        
        sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap=cmap, ax=ax,
                    xticklabels=CLASSES, yticklabels=CLASSES, 
                    cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100)
        
        for t in ax.texts: t.set_text(t.get_text() + " %")
            
        ax.set_title(title, fontweight='bold', fontsize=15)
        ax.set_ylabel('True Label', fontweight='bold', fontsize=12)
        ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=12)
        ax.tick_params(axis='x', rotation=45)

    plot_normalized_cm(labels_base, preds_base, axes[0], 'Centralized Baseline (Normalized)', 'Purples')
    plot_normalized_cm(labels_avg, preds_avg, axes[1], 'FedAvg Global Model (Normalized)', 'Blues')
    plot_normalized_cm(labels_prox, preds_prox, axes[2], 'FedProx Global Model (Normalized)', 'Greens')
    
    plt.tight_layout()
    plt.savefig("Ultimate_Normalized_Confusion_Matrices.png", dpi=300, bbox_inches='tight')
    print("\n✓ Boom! Normalized Plot saved successfully as 'Ultimate_Normalized_Confusion_Matrices.png'")
    plt.show()
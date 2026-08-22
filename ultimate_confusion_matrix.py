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
    try:
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval()
        print(f"✓ {model_name} loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading {model_path}: {e}")
        return None, None
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{model_name.replace(' ', '_')}_Report.csv")
    print(f"✓ {model_name} report saved as CSV.")
    
    return all_labels, all_preds

# --- 5. Run Evaluations (Using Ultimate weights) ---
import pandas as pd
labels_avg, preds_avg = evaluate_model("Ultimate_FedAvg_global_model.pth", "Ultimate FedAvg Global")
labels_prox, preds_prox = evaluate_model("Ultimate_FedProx_mu1.0_global_model.pth", "Ultimate FedProx Global mu1.0")
labels_base, preds_base = evaluate_model("Ultimate_Centralized_Baseline.pth", "Centralized Baseline (Ultimate)")

# --- 6. Plotting Ultimate Normalized Confusion Matrices ---
num_plots = 3 if labels_base is not None else 2
fig, axes = plt.subplots(1, num_plots, figsize=(8*num_plots, 7))

if num_plots == 2:
    axes = np.array([axes[0], axes[1]])

def plot_clean_normalized_cm(labels, preds, ax, title, cmap):
    cm = confusion_matrix(labels, preds, normalize='true') * 100 
    
    sns.heatmap(cm, annot=True, fmt='.0f', cmap=cmap, ax=ax,
                xticklabels=CLASSES, yticklabels=CLASSES, 
                cbar_kws={'label': 'Percentage (%)'}, vmin=0, vmax=100, annot_kws={"size": 13})
        
    ax.set_title(title, fontweight='bold', fontsize=16)
    ax.set_ylabel('True Label', fontweight='bold', fontsize=13)
    ax.set_xlabel('Predicted Label', fontweight='bold', fontsize=13)
    ax.tick_params(axis='x', rotation=45)

if labels_base is not None:
    plot_clean_normalized_cm(labels_base, preds_base, axes[0], 'Centralized Baseline', 'Purples')
    plot_clean_normalized_cm(labels_avg, preds_avg, axes[1], 'FedAvg Global (Ultimate)', 'Blues')
    plot_clean_normalized_cm(labels_prox, preds_prox, axes[2], 'FedProx Global (Ultimate) mu1.0', 'Greens')
else:
    plot_clean_normalized_cm(labels_avg, preds_avg, axes[0], 'FedAvg Global (Ultimate)', 'Blues')
    plot_clean_normalized_cm(labels_prox, preds_prox, axes[1], 'FedProx Global (Ultimate) mu1.0', 'Greens')

plt.tight_layout()
plt.savefig("Ultimate_Normalized_Confusion_Matrices_Clean.png", dpi=300, bbox_inches='tight')
print("\n✓ Boom! Clean and Ultimate plot saved successfully as 'Ultimate_Normalized_Confusion_Matrices_Clean.png'")
plt.show()
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import BloodMNIST
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_dataset = BloodMNIST(split="test", transform=test_transform, download=True, root="./data")
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

def get_resnet18():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8) 
    return model

def evaluate_model(model_path, model_name):
    model = get_resnet18().to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    report = classification_report(all_labels, all_preds, target_names=CLASSES, output_dict=True)
    pd.DataFrame(report).transpose().to_csv(f"{model_name.replace(' ', '_')}_Report.csv")
    return all_labels, all_preds

print("Evaluating 4 Ultimate models...")
l_base, p_base = evaluate_model("Ultimate_Centralized_Baseline.pth", "Centralized Baseline Ultimate")
l_avg, p_avg = evaluate_model("Ultimate_FedAvg_global_model.pth", "FedAvg Ultimate")
l_prox01, p_prox01 = evaluate_model("Ultimate_FedProx_mu0.1_global_model.pth", "FedProx mu0.1 Ultimate")
l_prox1, p_prox1 = evaluate_model("Ultimate_FedProx_mu1.0_global_model.pth", "FedProx mu1.0 Ultimate")

fig, axes = plt.subplots(2, 2, figsize=(16, 14))
axes = axes.flatten()

def plot_cm(labels, preds, ax, title, cmap):
    cm = confusion_matrix(labels, preds, normalize='true') * 100 
    sns.heatmap(cm, annot=True, fmt='.0f', cmap=cmap, ax=ax, xticklabels=CLASSES, yticklabels=CLASSES, vmin=0, vmax=100, annot_kws={"size": 12})
    ax.set_title(title, fontweight='bold', fontsize=14)
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

plot_cm(l_base, p_base, axes[0], 'Centralized Baseline', 'Purples')
plot_cm(l_avg, p_avg, axes[1], 'FedAvg Global', 'Blues')
plot_cm(l_prox01, p_prox01, axes[2], r'FedProx ($\mu=0.1$)', 'Oranges')
plot_cm(l_prox1, p_prox1, axes[3], r'FedProx ($\mu=1.0$)', 'Greens')

plt.tight_layout()
plt.savefig("Ultimate_Normalized_Confusion_Matrices_4Models.png", dpi=300, bbox_inches='tight')
print("✓ Saved Ultimate_Normalized_Confusion_Matrices_4Models.png")
plt.show()
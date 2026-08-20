import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from medmnist import BloodMNIST
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os

# --- 1. Setup & Device ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 
           'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

def get_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8) 
    return model

# --- 2. Load All 3 Models ---
model_baseline = get_model().to(DEVICE)
model_fedavg = get_model().to(DEVICE)
model_fedprox = get_model().to(DEVICE)

print("Loading model weights...")
try:
    model_baseline.load_state_dict(torch.load('resnet18_centralized.pth', map_location=DEVICE))
    model_fedavg.load_state_dict(torch.load('fedavg_global_model.pth', map_location=DEVICE))
    model_fedprox.load_state_dict(torch.load('fedprox_global_model.pth', map_location=DEVICE))
    
    model_baseline.eval()
    model_fedavg.eval()
    model_fedprox.eval()
    print("✓ All 3 models loaded successfully!")
except Exception as e:
    print(f"❌ Error loading models. Details: {e}")
    exit()

# --- 3. Prepare Dataset ---
print("Loading Test Set...")
dataset = BloodMNIST(split="test", download=True, root="./data")
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. Helper Function to Get Prediction & Confidence ---
def get_prediction_info(model, input_tensor):
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = F.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)
        return CLASSES[pred_idx.item()], conf.item() * 100

# --- 5. Generate Grad-CAM for Comparison ---
indices_to_test = [42, 100, 150] 

fig, axes = plt.subplots(len(indices_to_test), 4, figsize=(22, 16))

cam_baseline = GradCAM(model=model_baseline, target_layers=[model_baseline.layer4[-1]])
cam_fedavg = GradCAM(model=model_fedavg, target_layers=[model_fedavg.layer4[-1]])
cam_fedprox = GradCAM(model=model_fedprox, target_layers=[model_fedprox.layer4[-1]])

for i, idx in enumerate(indices_to_test):
    raw_image, label_array = dataset[idx]
    true_label = CLASSES[label_array[0]]
    
    resized_img = raw_image.resize((224, 224))
    rgb_img = np.float32(resized_img) / 255.0
    input_tensor = test_transform(raw_image).unsqueeze(0).to(DEVICE)
    
    # Get Predictions & Confidences
    pred_base, conf_base = get_prediction_info(model_baseline, input_tensor)
    pred_avg, conf_avg = get_prediction_info(model_fedavg, input_tensor)
    pred_prox, conf_prox = get_prediction_info(model_fedprox, input_tensor)
    
    # Generate Heatmaps
    heatmap_base = cam_baseline(input_tensor=input_tensor, targets=None)[0, :]
    heatmap_avg = cam_fedavg(input_tensor=input_tensor, targets=None)[0, :]
    heatmap_prox = cam_fedprox(input_tensor=input_tensor, targets=None)[0, :]
    
    vis_base = show_cam_on_image(rgb_img, heatmap_base, use_rgb=True)
    vis_avg = show_cam_on_image(rgb_img, heatmap_avg, use_rgb=True)
    vis_prox = show_cam_on_image(rgb_img, heatmap_prox, use_rgb=True)
    
    # Column 1: Original
    axes[i, 0].imshow(resized_img)
    axes[i, 0].set_title(f"Original\nTrue Label: {true_label}", fontweight='bold', fontsize=12)
    axes[i, 0].axis('off')
    
    # Column 2: Centralized
    axes[i, 1].imshow(vis_base)
    color_base = 'darkgreen' if pred_base == true_label else 'darkred'
    axes[i, 1].set_title(f"Centralized Baseline\nPred: {pred_base} ({conf_base:.1f}%)", fontweight='bold', fontsize=11, color=color_base)
    axes[i, 1].axis('off')

    # Column 3: FedAvg
    axes[i, 2].imshow(vis_avg)
    color_avg = 'darkgreen' if pred_avg == true_label else 'darkred'
    axes[i, 2].set_title(f"FedAvg Global\nPred: {pred_avg} ({conf_avg:.1f}%)", fontweight='bold', fontsize=11, color=color_avg)
    axes[i, 2].axis('off')
    
    # Column 4: FedProx
    axes[i, 3].imshow(vis_prox)
    color_prox = 'darkgreen' if pred_prox == true_label else 'darkred'
    axes[i, 3].set_title(f"FedProx Global\nPred: {pred_prox} ({conf_prox:.1f}%)", fontweight='bold', fontsize=11, color=color_prox)
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig("Scientific_GradCAM_Comparison.png", dpi=300, bbox_inches='tight')
print("\n✓ Boom! The scientific Grad-CAM plot saved as 'Scientific_GradCAM_Comparison.png'")
plt.show()
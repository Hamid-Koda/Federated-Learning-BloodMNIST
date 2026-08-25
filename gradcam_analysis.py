import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from medmnist import BloodMNIST
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

def get_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

print("Loading 4 Ultimate models...")
m_base = get_model('Ultimate_Centralized_Baseline.pth')
m_avg = get_model('Ultimate_FedAvg_global_model.pth')
m_prox01 = get_model('Ultimate_FedProx_mu0.1_global_model.pth')
m_prox1 = get_model('Ultimate_FedProx_mu1.0_global_model.pth')

dataset = BloodMNIST(split="test", download=True, root="./data")
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def get_pred(model, tensor):
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)
        conf, idx = torch.max(probs, 1)
        return CLASSES[idx.item()], conf.item() * 100

indices_to_test = [42, 100, 150] 
fig, axes = plt.subplots(len(indices_to_test), 5, figsize=(25, 14))

cams = {
    'Base': GradCAM(model=m_base, target_layers=[m_base.layer4[-1]]),
    'Avg': GradCAM(model=m_avg, target_layers=[m_avg.layer4[-1]]),
    'Prox01': GradCAM(model=m_prox01, target_layers=[m_prox01.layer4[-1]]),
    'Prox1': GradCAM(model=m_prox1, target_layers=[m_prox1.layer4[-1]])
}

for i, idx in enumerate(indices_to_test):
    raw_image, label_array = dataset[idx]
    true_label = CLASSES[label_array[0]]
    
    resized_img = raw_image.resize((224, 224))
    rgb_img = np.float32(resized_img) / 255.0
    input_tensor = test_transform(raw_image).unsqueeze(0).to(DEVICE)
    
    axes[i, 0].imshow(resized_img)
    axes[i, 0].set_title(f"Original\nTrue Label: {true_label}", fontweight='bold')
    axes[i, 0].axis('off')
    
    models_dict = [
        (m_base, cams['Base'], 'Centralized'),
        (m_avg, cams['Avg'], 'FedAvg'),
        (m_prox01, cams['Prox01'], r'FedProx ($\mu=0.1$)'),
        (m_prox1, cams['Prox1'], r'FedProx ($\mu=1.0$)')
    ]
    
    for j, (mod, cam, title) in enumerate(models_dict, start=1):
        pred, conf = get_pred(mod, input_tensor)
        heatmap = cam(input_tensor=input_tensor, targets=None)[0, :]
        vis = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
        
        axes[i, j].imshow(vis)
        color = 'darkgreen' if pred == true_label else 'darkred'
        axes[i, j].set_title(f"{title}\nPred: {pred} ({conf:.1f}%)", fontweight='bold', color=color)
        axes[i, j].axis('off')

plt.tight_layout()
plt.savefig("Ultimate_Scientific_GradCAM_4Models.png", dpi=300, bbox_inches='tight')
print("✓ Saved Ultimate_Scientific_GradCAM_4Models.png")
plt.show()
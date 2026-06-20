import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms, models
from medmnist import BloodMNIST
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# 1. Device Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Load the pre-trained ResNet18 weights
print("Loading the trained ResNet18 model weights...")
model = models.resnet18(weights=None)
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 8) 
try:
    model.load_state_dict(torch.load('resnet18_blood_model.pth', map_location=DEVICE))
    model.eval()
    model = model.to(DEVICE)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# 3. Load Dataset & Prepare Transform
print("Fetching multiple blood cell images...")
dataset = BloodMNIST(split="test", download=True, root="./data")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 4. Select the same multiple diverse images for fair comparison
indices_to_test = [15, 42, 100]

# Setup Plot
fig, axes = plt.subplots(len(indices_to_test), 2, figsize=(10, 12))

target_layers = [model.layer4[-1]]
cam = GradCAM(model=model, target_layers=target_layers)

for i, idx in enumerate(indices_to_test):
    raw_image, label = dataset[idx]
    
    resized_img = raw_image.resize((224, 224))
    rgb_img = np.float32(resized_img) / 255.0
    
    input_tensor = test_transform(raw_image).unsqueeze(0).to(DEVICE)
    
    # Generate Heatmap
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    
    # Plot Original
    axes[i, 0].imshow(resized_img)
    axes[i, 0].set_title(f"Original Image (Class ID: {label[0]})", fontweight='bold')
    axes[i, 0].axis('off')
    
    # Plot Heatmap
    axes[i, 1].imshow(visualization)
    axes[i, 1].set_title("ResNet18 Grad-CAM Focus", fontweight='bold')
    axes[i, 1].axis('off')

plt.tight_layout()

# Save the multi-image plot
plot_filename = "gradcam_resnet18_multiple.png"
plt.savefig(plot_filename, dpi=300)
print(f"\n✓ Success! Plot saved automatically as '{plot_filename}'")
plt.show()
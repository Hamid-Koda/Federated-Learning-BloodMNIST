import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# --- 1. Setup ---
DEVICE = torch.device("cpu") 
CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 
           'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

# --- 2. Load Ultimate Models ---
def get_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8) 
    return model.to(DEVICE)

try:
    model_base = get_model()
    model_base.load_state_dict(torch.load('Ultimate_Centralized_Baseline.pth', map_location=DEVICE))
    model_base.eval()

    model_avg = get_model()
    model_avg.load_state_dict(torch.load('Ultimate_FedAvg_global_model.pth', map_location=DEVICE))
    model_avg.eval()

    model_prox = get_model()
    model_prox.load_state_dict(torch.load('Ultimate_FedProx_mu1.0_global_model.pth', map_location=DEVICE))
    model_prox.eval()
    print("✓ Models loaded successfully for UI!")
except Exception as e:
    print(f"❌ Error loading models. Make sure .pth files are in the same folder. Details: {e}")

# --- 3. Transformations ---
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 4. Prediction & XAI Function ---
def analyze_blood_cell(image):
    # Preprocess
    img_pil = Image.fromarray(image).convert('RGB')
    input_tensor = test_transform(img_pil).unsqueeze(0).to(DEVICE)
    rgb_img = np.float32(img_pil.resize((224, 224))) / 255.0

    def get_cam_and_pred(model):
        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            pred_class = CLASSES[pred_idx.item()]
            conf_score = conf.item() * 100
            
        # Grad-CAM
        cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
        heatmap = cam(input_tensor=input_tensor, targets=None)[0, :]
        vis = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
        return f"{pred_class} ({conf_score:.1f}%)", vis

    pred_base, img_base = get_cam_and_pred(model_base)
    pred_avg, img_avg = get_cam_and_pred(model_avg)
    pred_prox, img_prox = get_cam_and_pred(model_prox)

    return pred_base, img_base, pred_avg, img_avg, pred_prox, img_prox

# --- 5. Gradio Interface ---
with gr.Blocks(theme=gr.themes.Soft()) as interface:
    gr.Markdown("# 🔬 Blood Cell Classification & Federated Learning Analysis")
    gr.Markdown("Upload a microscopic blood cell image to see how Centralized Learning compares to Federated Learning (FedAvg vs. FedProx) under Non-IID conditions.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Blood Cell Image")
            analyze_btn = gr.Button("Analyze Cell", variant="primary")
            
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Centralized Baseline")
                    text_base = gr.Textbox(label="Prediction & Confidence")
                    out_img_base = gr.Image(label="Grad-CAM Focus")
                with gr.Column():
                    gr.Markdown("### FedAvg (Fails in Non-IID)")
                    text_avg = gr.Textbox(label="Prediction & Confidence")
                    out_img_avg = gr.Image(label="Grad-CAM Focus")
                with gr.Column():
                    gr.Markdown("### FedProx (Recovers in Non-IID)")
                    text_prox = gr.Textbox(label="Prediction & Confidence")
                    out_img_prox = gr.Image(label="Grad-CAM Focus")

    analyze_btn.click(
        fn=analyze_blood_cell,
        inputs=input_image,
        outputs=[text_base, out_img_base, text_avg, out_img_avg, text_prox, out_img_prox]
    )

if __name__ == "__main__":
    interface.launch(share=False)
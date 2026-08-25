import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import os
import time
import threading

# --- 1. Setup ---
DEVICE = torch.device("cpu")
CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 
           'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

def get_model(path):
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 8) 
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

# --- 2. Load Models ---
try:
    model_base = get_model('Ultimate_Centralized_Baseline.pth')
    model_avg = get_model('Ultimate_FedAvg_global_model.pth')
    model_prox01 = get_model('Ultimate_FedProx_mu0.1_global_model.pth')
    model_prox1 = get_model('Ultimate_FedProx_mu1.0_global_model.pth')
    print("✓ All 4 Models loaded successfully for UI!")
except Exception as e:
    print(f" Error loading models. Details: {e}")

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- 3. Functions ---
def analyze_blood_cell(image):
    if image is None:
        return ["Please upload an image!"] * 8
        
    img_pil = Image.fromarray(image).convert('RGB')
    input_tensor = test_transform(img_pil).unsqueeze(0).to(DEVICE)
    rgb_img = np.float32(img_pil.resize((224, 224))) / 255.0

    def get_cam_and_pred(model):
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            conf, pred_idx = torch.max(probs, 1)
            pred_class = CLASSES[pred_idx.item()]
            conf_score = conf.item() * 100
            
        cam = GradCAM(model=model, target_layers=[model.layer4[-1]])
        heatmap = cam(input_tensor=input_tensor, targets=None)[0, :]
        vis = show_cam_on_image(rgb_img, heatmap, use_rgb=True)
        return f"{pred_class} ({conf_score:.1f}%)", vis

    pred_base, img_base = get_cam_and_pred(model_base)
    pred_avg, img_avg = get_cam_and_pred(model_avg)
    pred_prox01, img_prox01 = get_cam_and_pred(model_prox01)
    pred_prox1, img_prox1 = get_cam_and_pred(model_prox1)

    return pred_base, img_base, pred_avg, img_avg, pred_prox01, img_prox01, pred_prox1, img_prox1

def stop_app():
    def kill_process():
        time.sleep(1.5) 
        os._exit(0)
    threading.Thread(target=kill_process).start()
    return gr.update(value="🛑 Server Stopped! You can close this tab.", interactive=False, variant="secondary")

# --- 4. Gradio Interface (Beautiful 2x2 Layout) ---
with gr.Blocks(theme=gr.themes.Soft(primary_hue="indigo")) as interface:
    gr.Markdown("<h1 style='text-align: center;'>🔬 Ultimate Blood Cell Classification & FL Analysis</h1>")
    gr.Markdown("<p style='text-align: center; font-size: 16px;'>Upload a microscopic blood cell image to compare Centralized Learning with FedAvg and FedProx.</p>")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(label="Upload Blood Cell Image", height=220)
            analyze_btn = gr.Button("🔍 Analyze Cell", variant="primary", size="lg")
            stop_btn = gr.Button("🛑 Stop & Exit Server", variant="stop", size="lg")
            
        with gr.Column(scale=3):
            with gr.Row():
                with gr.Group():
                    gr.Markdown("<h3 style='text-align: center; color: #4338ca;'>1. Central Baseline</h3>")
                    text_base = gr.Textbox(label="Prediction & Confidence")
                    out_img_base = gr.Image(label="Grad-CAM Focus", show_label=False)
                    
                with gr.Group():
                    gr.Markdown("<h3 style='text-align: center; color: #b91c1c;'>2. FedAvg (Fails in Non-IID)</h3>")
                    text_avg = gr.Textbox(label="Prediction & Confidence")
                    out_img_avg = gr.Image(label="Grad-CAM Focus", show_label=False)
                    
            with gr.Row():
                with gr.Group():
                    gr.Markdown("<h3 style='text-align: center; color: #c2410c;'>3. FedProx (μ=0.1)</h3>")
                    text_prox01 = gr.Textbox(label="Prediction & Confidence")
                    out_img_prox01 = gr.Image(label="Grad-CAM Focus", show_label=False)
                    
                with gr.Group():
                    gr.Markdown("<h3 style='text-align: center; color: #15803d;'>4. FedProx (μ=1.0)</h3>")
                    text_prox1 = gr.Textbox(label="Prediction & Confidence")
                    out_img_prox1 = gr.Image(label="Grad-CAM Focus", show_label=False)

    analyze_btn.click(
        fn=analyze_blood_cell,
        inputs=input_image,
        outputs=[text_base, out_img_base, text_avg, out_img_avg, text_prox01, out_img_prox01, text_prox1, out_img_prox1]
    )
    
    stop_btn.click(
        fn=stop_app, 
        inputs=None, 
        outputs=stop_btn,
        js="() => { setTimeout(function(){ window.close(); }, 1500); }"
    )

if __name__ == "__main__":
    interface.launch(share=False)
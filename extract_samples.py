import os
from medmnist import BloodMNIST
from PIL import Image

CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 
           'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

print("Loading BloodMNIST Test Set...")
dataset = BloodMNIST(split="test", download=True, root="./data")

output_dir = "UI_Test_Samples"
os.makedirs(output_dir, exist_ok=True)

samples_per_class = 3
saved_counts = {c: 0 for c in CLASSES}

print(f"Extracting {samples_per_class} images per class...")

for img, label_array in dataset:
    class_idx = label_array[0]
    class_name = CLASSES[class_idx]
    
    if saved_counts[class_name] < samples_per_class:
        count = saved_counts[class_name] + 1
        file_name = f"{class_name}_TrueLabel_{count}.png"
        img.save(os.path.join(output_dir, file_name))
        saved_counts[class_name] += 1
        
    if all(count >= samples_per_class for count in saved_counts.values()):
        break

print(f"✓ Done! Check the '{output_dir}' folder in your project directory.")
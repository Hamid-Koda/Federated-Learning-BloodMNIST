import matplotlib.pyplot as plt
import numpy as np
import torch
from medmnist import BloodMNIST

train_dataset = BloodMNIST(split='train', download=True, root='./data')

print(f"Dataset length: {len(train_dataset)}")
classes = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']

plt.figure(figsize=(12, 6))
for i in range(8):
    idx = np.random.randint(0, len(train_dataset))
    img, target = train_dataset[idx]
    
    plt.subplot(2, 4, i + 1)
    plt.imshow(img)
    plt.title(f"Class {target[0]}\n{classes[target[0]]}")
    plt.axis('off')

plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from medmnist import BloodMNIST
import pandas as pd
import os

# --- 1. Setup ---
CLASSES = ['Basophil', 'Eosinophil', 'Erythroblast', 'IG', 
           'Lymphocyte', 'Monocyte', 'Neutrophil', 'Platelet']
NUM_CLIENTS = 3

# --- 2. Load Dataset ---
print("Loading Training Dataset...")
os.makedirs("./data", exist_ok=True)
train_dataset = BloodMNIST(split="train", download=True, root="./data")

# --- 3. Apply the exact same Non-IID logic ---
total_len = len(train_dataset)
labels = np.array([target[0] for _, target in train_dataset])
sorted_indices = np.argsort(labels)

base_len = total_len // NUM_CLIENTS
indices_list = []
for i in range(NUM_CLIENTS):
    start_idx = i * base_len
    end_idx = (i + 1) * base_len if i < NUM_CLIENTS - 1 else total_len
    indices_list.append(sorted_indices[start_idx:end_idx])

# --- 4. Count Class Distribution per Client ---
distribution = np.zeros((NUM_CLIENTS, len(CLASSES)), dtype=int)

for client_id, indices in enumerate(indices_list):
    client_labels = labels[indices]
    unique, counts = np.unique(client_labels, return_counts=True)
    for cls_idx, count in zip(unique, counts):
        distribution[client_id, cls_idx] = count

# --- 5. Plotting the Distribution ---
df = pd.DataFrame(distribution, columns=CLASSES, index=[f'Client {i+1}' for i in range(NUM_CLIENTS)])
print("\n--- Data Distribution Matrix ---")
print(df)

# Plot
ax = df.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='tab20')
plt.title('Non-IID Data Distribution Across Federated Clients', fontweight='bold', fontsize=14)
plt.xlabel('Clients', fontweight='bold')
plt.ylabel('Number of Images', fontweight='bold')
plt.xticks(rotation=0)
plt.legend(title='Blood Cell Classes', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("Client_Data_Distribution.png", dpi=300, bbox_inches='tight')
print("\n✓ Success! Plot saved as 'Client_Data_Distribution.png'")
plt.show()
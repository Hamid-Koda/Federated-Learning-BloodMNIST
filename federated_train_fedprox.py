import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import BloodMNIST
import matplotlib.pyplot as plt
import numpy as np
import os

# --- 1. Hyperparameters & Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  
EPOCHS = 2       
NUM_CLIENTS = 3
NUM_ROUNDS = 30  

# --- 2. Data Preparation ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Downloading and preparing dataset...")
os.makedirs("./data", exist_ok=True)
dataset = BloodMNIST(split="train", transform=transform, download=True, root="./data")

# --- Non-IID Partitioning (FULL DATASET) ---
total_len = len(dataset)
labels = np.array([target[0] for _, target in dataset])
sorted_indices = np.argsort(labels)

base_len = total_len // NUM_CLIENTS
indices_list = []
for i in range(NUM_CLIENTS):
    start_idx = i * base_len
    end_idx = (i + 1) * base_len if i < NUM_CLIENTS - 1 else total_len
    indices_list.append(sorted_indices[start_idx:end_idx])

datasets = [torch.utils.data.Subset(dataset, idx) for idx in indices_list]
print(f"Non-IID partitioning completed for {NUM_CLIENTS} clients with FULL dataset ({total_len} images).")

# --- 3. Federated Training Logic ---
def train(model, loader, global_parameters, cid, mu=1.0):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    
    params_dict = zip(model.state_dict().keys(), global_parameters)
    global_state_dict = {k: torch.tensor(v).to(DEVICE) for k, v in params_dict}
    
    model.train()
    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        proximal_term = 0.0
        for name, local_param in model.named_parameters():
            if local_param.requires_grad:
                global_param = global_state_dict[name]
                proximal_term += torch.square((local_param - global_param).norm(2))
        
        loss += (mu / 2) * proximal_term
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print(f" [Heartbeat] Client {cid} - Batch {batch_idx}/{len(loader)} - Loss: {loss.item():.4f}")

# --- 4. Flower Client Setup ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, client_dataset, cid):
        self.cid = cid
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.fc = nn.Linear(self.model.fc.in_features, 8)
        self.model = self.model.to(DEVICE)
        self.loader = DataLoader(client_dataset, batch_size=BATCH_SIZE, shuffle=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k, v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        print(f"\n >>> Client {self.cid} starting local training on {DEVICE}...")
        self.set_parameters(parameters)
        train(self.model, self.loader, parameters, cid=self.cid, mu=1.0)
        print(f" <<< Client {self.cid} finished training.")
        return self.get_parameters(config={}), len(self.loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        criterion = nn.CrossEntropyLoss()
        self.model.eval()
        loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in self.loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
                outputs = self.model(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs, 1)[1] == labels).sum().item()
        
        accuracy = correct / len(self.loader.dataset)
        return float(loss / len(self.loader)), len(self.loader.dataset), {"accuracy": float(accuracy)}

# --- 5. Server Strategy & Simulation ---
def aggregate_metrics(metrics):
    accuracies = [m[1]["accuracy"] for m in metrics]
    examples = [m[0] for m in metrics]
    weighted_avg = sum(a * e for a, e in zip(accuracies, examples)) / sum(examples)
    return {"accuracy": weighted_avg}

strategy = fl.server.strategy.FedProx(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
    evaluate_metrics_aggregation_fn=aggregate_metrics,
    proximal_mu=1.0
)

def client_fn(cid):
    return FlowerClient(datasets[int(cid)], cid)

print("\n" + "="*40)
print("Starting ResNet18 + FedProx Simulation on Google Colab")
print("="*40)

results = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_gpus": 1, "num_cpus": 2}, 
)

# --- 6. Plotting Results ---
rounds_loss, losses = zip(*results.losses_distributed)
rounds_acc, accuracies = zip(*results.metrics_distributed["accuracy"])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(rounds_loss, losses, marker='o', color='crimson', linewidth=2, label='Distributed Loss')
ax1.set_title('ResNet18 FedProx Global Loss', fontsize=12, fontweight='bold')
ax1.set_xlabel('Federated Rounds')
ax1.set_ylabel('Loss')
ax1.grid(True, linestyle='--', alpha=0.5)
ax1.legend()

ax2.plot(rounds_acc, [a * 100 for a in accuracies], marker='s', color='teal', linewidth=2, label='Distributed Accuracy')
ax2.set_title('ResNet18 FedProx Global Accuracy', fontsize=12, fontweight='bold')
ax2.set_xlabel('Federated Rounds')
ax2.set_ylabel('Accuracy (%)')
ax2.grid(True, linestyle='--', alpha=0.5)
ax2.legend()

plt.tight_layout()
plt.savefig("federated_metrics_FedProx_ResNet18_Colab.png", dpi=300)
print(f"\n✓ Success! Plot generated.")
plt.show()
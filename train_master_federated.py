import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms, models
from medmnist import BloodMNIST
import flwr as fl
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import OrderedDict


# --- 0. Set Seeds for Reproducibility ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# --- 1. Hyperparameters ---
NUM_CLIENTS = 3
NUM_ROUNDS = 50
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🚀 Running on Device: {DEVICE}")

# --- 2. Data Preparation (with Updated Realistic Augmentation) ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20), 
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

os.makedirs("./data", exist_ok=True)
train_dataset = BloodMNIST(split="train", transform=train_transform, download=True, root="./data")
test_dataset = BloodMNIST(split="test", transform=test_transform, download=True, root="./data")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Strict Non-IID Splitting Logic
labels = np.array([target[0] for _, target in train_dataset])
sorted_indices = np.argsort(labels)
base_len = len(train_dataset) // NUM_CLIENTS
client_datasets = []

for i in range(NUM_CLIENTS):
    start_idx = i * base_len
    end_idx = (i + 1) * base_len if i < NUM_CLIENTS - 1 else len(train_dataset)
    client_datasets.append(Subset(train_dataset, sorted_indices[start_idx:end_idx]))

# --- 3. Model Architecture ---
def get_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Linear(model.fc.in_features, 8)
    return model.to(DEVICE)

# --- 4. Client Definition (Unified for FedAvg & FedProx) ---
class BloodClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, mu):
        self.model = model
        self.train_loader = train_loader
        self.mu = mu

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        global_params = [p.clone().detach() for p in self.model.parameters()]

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)

        self.model.train()
        for epoch in range(LOCAL_EPOCHS):
            for images, labels in self.train_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
                
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)

                if self.mu > 0.0:
                    proximal_term = 0.0
                    for local_param, global_param in zip(self.model.parameters(), global_params):
                        proximal_term += torch.square((local_param - global_param).norm(2))
                    loss += (self.mu / 2) * proximal_term

                loss.backward()
                optimizer.step()

        return self.get_parameters(config=None), len(self.train_loader.dataset), {}

# --- 5. Server Evaluation Function ---
def get_evaluate_fn(global_model):
    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, fl.common.Scalar]):
        params_dict = zip(global_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        global_model.load_state_dict(state_dict, strict=True)

        global_model.eval()
        criterion = nn.CrossEntropyLoss()
        loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
                outputs = global_model(images)
                loss += criterion(outputs, labels).item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        return loss / total, {"accuracy": accuracy}
    return evaluate

# --- 6. The Master Runner Function ---
def run_experiment(experiment_name: str, mu_value: float):
    print(f"\n{'='*60}")
    print(f"🔥 STARTING EXPERIMENT: {experiment_name} (mu = {mu_value})")
    print(f"{'='*60}\n")

    global_model = get_resnet18()

    strategy = fl.server.strategy.FedAvg(
        fraction_fit=1.0,
        fraction_evaluate=0.0,
        min_fit_clients=NUM_CLIENTS,
        min_available_clients=NUM_CLIENTS,
        evaluate_fn=get_evaluate_fn(global_model),
    )

    def client_fn(cid: str):
        train_loader = DataLoader(client_datasets[int(cid)], batch_size=BATCH_SIZE, shuffle=True)
        return BloodClient(get_resnet18(), train_loader, mu=mu_value)

    results = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
        strategy=strategy,
        client_resources={"num_cpus": 1, "num_gpus": 1 if torch.cuda.is_available() else 0},
    )

    rounds = [res[0] for res in results.metrics_centralized["accuracy"]]
    accs = [res[1] for res in results.metrics_centralized["accuracy"]]
    losses = [res[1] for res in results.losses_centralized]

    df = pd.DataFrame({"Round": rounds, "Loss": losses, "Accuracy": accs})
    csv_filename = f"{experiment_name}_history.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n✅ Training Logs saved to: {csv_filename}")

    weight_filename = f"{experiment_name}_global_model.pth"
    torch.save(global_model.state_dict(), weight_filename)
    print(f"✅ Final Model Weights saved to: {weight_filename}\n")

# --- 7. EXECUTION ---
if __name__ == "__main__":
    run_experiment("Ultimate_FedAvg", mu_value=0.0)

    run_experiment("Ultimate_FedProx_mu1.0", mu_value=1.0)
    
    run_experiment("Ultimate_FedProx_mu0.1", mu_value=0.1)
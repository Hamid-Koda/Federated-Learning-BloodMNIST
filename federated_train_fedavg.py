import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models
from medmnist import BloodMNIST
import numpy as np
from collections import OrderedDict
import os

# --- 1. Hyperparameters & Device Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
LOCAL_EPOCHS = 3       # Loop bug fixed!
NUM_CLIENTS = 3
NUM_ROUNDS = 50         # Set to 2 for laptop testing. Change to 30-50 on Colab/Kaggle.

# --- 2. Data Preparation ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(90),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Loading Datasets...")
os.makedirs("./data", exist_ok=True)
train_dataset = BloodMNIST(split="train", transform=train_transform, download=True, root="./data")
test_dataset = BloodMNIST(split="test", transform=test_transform, download=True, root="./data")

# The crucial global test loader!
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Non-IID Partitioning ---
total_len = len(train_dataset)
labels = np.array([target[0] for _, target in train_dataset])
sorted_indices = np.argsort(labels)

base_len = total_len // NUM_CLIENTS
indices_list = []
for i in range(NUM_CLIENTS):
    start_idx = i * base_len
    end_idx = (i + 1) * base_len if i < NUM_CLIENTS - 1 else total_len
    indices_list.append(sorted_indices[start_idx:end_idx])

datasets = [torch.utils.data.Subset(train_dataset, idx) for idx in indices_list]

# --- 4. Model Setup (ResNet18 with fine-tuning) ---
def get_resnet18():
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Freeze all layers first
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze Layer 4 and fully connected layer for specialized learning
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 8) 
    return model

def set_parameters(model, parameters):
    params_dict = zip(model.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)

# --- 5. Flower Client ---
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, dataset, cid):
        self.cid = cid
        self.model = get_resnet18().to(DEVICE)
        self.loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def fit(self, parameters, config):
        set_parameters(self.model, parameters)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=0.001)
        
        self.model.train()
        # FIX: The missing epoch loop is now here!
        for epoch in range(LOCAL_EPOCHS):
            for images, labels in self.loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
        return self.get_parameters(config={}), len(self.loader.dataset), {}

def client_fn(cid):
    return FlowerClient(datasets[int(cid)], cid)

# --- 6. Global Evaluation & Saving ---
def get_evaluate_fn(test_loader):
    def evaluate(server_round, parameters, config):
        model = get_resnet18().to(DEVICE)
        set_parameters(model, parameters)
        model.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE).long().squeeze()
                outputs = model(images)
                loss += criterion(outputs, labels).item()
                correct += (torch.max(outputs, 1)[1] == labels).sum().item()
                
        accuracy = correct / len(test_loader.dataset)
        
        # Save the model exactly at the last round
        if server_round == NUM_ROUNDS:
            torch.save(model.state_dict(), "fedavg_global_model.pth")
            print(f"\n[Round {server_round}] ✓ Global model saved successfully as 'fedavg_global_model.pth'")
            
        return float(loss / len(test_loader)), {"accuracy": float(accuracy)}
    return evaluate

# --- 7. Server Strategy & Simulation ---
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0,
    min_fit_clients=NUM_CLIENTS,
    min_available_clients=NUM_CLIENTS,
    evaluate_fn=get_evaluate_fn(test_loader) # Centralized Test Set Evaluation!
)

print("\n" + "="*40)
print(f"Starting FedAvg ResNet18 on {DEVICE}")
print("="*40)

results = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS),
    strategy=strategy,
    client_resources={"num_cpus": 2, "num_gpus": 1 if torch.cuda.is_available() else 0}
)
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from medmnist import BloodMNIST
import matplotlib.pyplot as plt  

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 64
EPOCHS = 1 
NUM_CLIENTS = 3

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = BloodMNIST(split="train", transform=transform, download=True, root="./data")

total_len = len(dataset)
base_len = total_len // NUM_CLIENTS
lengths = [base_len] * NUM_CLIENTS
lengths[-1] += total_len - sum(lengths)

g = torch.Generator().manual_seed(42)
datasets = random_split(dataset, lengths, generator=g)

print(f"Total dataset size: {total_len}")
print(f"Dataset sizes per client: {lengths}")

class BloodCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16,32,3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*7*7,128),
            nn.ReLU(),
            nn.Linear(128,8)
        )

    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=x.view(x.size(0),-1)
        x=self.fc(x)
        return x

def train(model,loader):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    for images,labels in loader:
        images,labels=images.to(DEVICE),labels.to(DEVICE).long().squeeze()
        outputs=model(images)
        loss=criterion(outputs,labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, dataset):
        self.model = BloodCNN().to(DEVICE)
        self.loader = DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True)

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = {k: torch.tensor(v) for k,v in params_dict}
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(self.model,self.loader)
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

def aggregate_metrics(metrics):
    accuracies = [m[1]["accuracy"] for m in metrics]
    examples = [m[0] for m in metrics]
    weighted_avg = sum(a * e for a, e in zip(accuracies, examples)) / sum(examples)
    return {"accuracy": weighted_avg}

strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=aggregate_metrics,
)

def client_fn(cid):
    dataset = datasets[int(cid)]
    return FlowerClient(dataset)

results = fl.simulation.start_simulation(
    client_fn=client_fn,
    num_clients=NUM_CLIENTS,
    config=fl.server.ServerConfig(num_rounds=5),
    strategy=strategy,
    ray_init_args={"object_store_memory": 100 * 1024 * 1024}
)

print("\n" + "="*30)
print("Generating Federated Learning Plots...")
print("="*30)

try:
    rounds_loss, losses = zip(*results.losses_distributed)
    rounds_acc, accuracies = zip(*results.metrics_distributed["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(rounds_loss, losses, marker='o', color='crimson', linewidth=2, label='Distributed Loss')
    ax1.set_title('Global Model Loss over Rounds', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Federated Rounds')
    ax1.set_ylabel('Loss')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    ax2.plot(rounds_acc, [a * 100 for a in accuracies], marker='s', color='teal', linewidth=2, label='Distributed Accuracy')
    ax2.set_title('Global Model Accuracy over Rounds', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Federated Rounds')
    ax2.set_ylabel('Accuracy (%)')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    
    plot_filename = "federated_metrics_plot.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"✓ Success! Plot saved automatically as '{plot_filename}'")
    
    plt.show()

except Exception as e:
    print(f"⚠️ An error occurred while plotting: {e}")
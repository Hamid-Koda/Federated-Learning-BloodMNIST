import pandas as pd
import matplotlib.pyplot as plt

# --- 1. Load Automated Logs (CSV Files) ---
try:
    df_avg = pd.read_csv("Ultimate_FedAvg_history.csv")
    df_prox_1 = pd.read_csv("Ultimate_FedProx_mu1.0_history.csv")
    df_prox_01 = pd.read_csv("Ultimate_FedProx_mu0.1_history.csv")
    df_base = pd.read_csv("Ultimate_Centralized_history.csv") 
    print("✓ All CSV logs loaded successfully!")
except FileNotFoundError as e:
    print(f"❌ Error loading CSV files: {e}")
    exit()

# Extract Data
rounds = df_avg['Round'].tolist()
acc_avg = df_avg['Accuracy'] * 100
acc_prox_1 = df_prox_1['Accuracy'] * 100
acc_prox_01 = df_prox_01['Accuracy'] * 100

loss_avg = df_avg['Loss']
loss_prox_1 = df_prox_1['Loss']
loss_prox_01 = df_prox_01['Loss']

CENTRALIZED_MAX_ACC = df_base['Accuracy'].max() * 100

# --- 2. Plotting the Ultimate Convergence Curves ---
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

# Plot 1: Test Accuracy vs Rounds
ax1.plot(rounds, acc_avg, label='FedAvg Global', color='#d62728', linestyle='--', marker='o', markersize=4, alpha=0.85)
ax1.plot(rounds, acc_prox_01, label=r'FedProx ($\mu=0.1$) [Weak Penalty]', color='#ff7f0e', linestyle='-.', marker='^', markersize=4)
ax1.plot(rounds, acc_prox_1, label=r'FedProx ($\mu=1.0$) [Optimal Penalty]', color='#2ca02c', linewidth=2.5, marker='s', markersize=4)

ax1.axhline(y=CENTRALIZED_MAX_ACC, color='#1f77b4', linestyle=':', linewidth=2.5, label=f'Centralized Baseline Peak ({CENTRALIZED_MAX_ACC}%)')

ax1.set_title('Global Model Test Accuracy over Federated Rounds', fontsize=14, fontweight='bold')
ax1.set_xlabel('Federated Communication Rounds', fontsize=12, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax1.set_ylim([0, 105])
ax1.legend(loc='lower right', frameon=True, fontsize=11)
ax1.grid(True, linestyle='--', alpha=0.7)

# Plot 2: Test Loss vs Rounds
ax2.plot(rounds, loss_avg, label='FedAvg Global', color='#d62728', linestyle='--', marker='o', markersize=4, alpha=0.85)
ax2.plot(rounds, loss_prox_01, label=r'FedProx ($\mu=0.1$)', color='#ff7f0e', linestyle='-.', marker='^', markersize=4)
ax2.plot(rounds, loss_prox_1, label=r'FedProx ($\mu=1.0$)', color='#2ca02c', linewidth=2.5, marker='s', markersize=4)

ax2.set_title('Global Model Test Loss over Federated Rounds', fontsize=14, fontweight='bold')
ax2.set_xlabel('Federated Communication Rounds', fontsize=12, fontweight='bold')
ax2.set_ylabel('Cross-Entropy Loss', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right', frameon=True, fontsize=11)
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig("Ultimate_Convergence_with_Ablation.png", dpi=300, bbox_inches='tight')
print("\n✓ Success! The ultimate convergence plot saved as 'Ultimate_Convergence_with_Ablation.png'")
plt.show()
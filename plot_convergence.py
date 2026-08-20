import matplotlib.pyplot as plt
import numpy as np

# --- 1. Data Extracted from Logs ---
rounds = list(range(0, 51))

# FedAvg Logs
fedavg_loss = [
    2.2677, 1.9828, 2.0614, 1.7026, 1.6558, 1.8740, 1.7870, 1.6095, 1.6763, 1.5584,
    1.5767, 1.5702, 1.5612, 1.5028, 1.4700, 1.3896, 1.4068, 1.4131, 1.3565, 1.5219,
    1.3605, 1.4436, 1.3286, 1.4711, 1.3410, 1.5368, 1.3315, 1.2483, 1.0948, 1.1643,
    1.2449, 1.3128, 1.6395, 1.7296, 1.5128, 1.5472, 1.6203, 1.4160, 1.4990, 1.4944,
    1.2475, 1.4379, 1.3627, 1.2075, 1.1087, 1.2924, 1.0188, 1.2406, 1.0939, 1.1018, 1.0129
]

fedavg_acc = [
    0.1567, 0.1114, 0.1485, 0.3587, 0.5016, 0.3619, 0.3841, 0.4005, 0.3973, 0.4212,
    0.4095, 0.5931, 0.4315, 0.5592, 0.4151, 0.6080, 0.5379, 0.5767, 0.5647, 0.5118,
    0.5259, 0.5618, 0.5472, 0.5703, 0.5115, 0.4458, 0.4323, 0.4373, 0.5177, 0.4426,
    0.4958, 0.5153, 0.4265, 0.4811, 0.5048, 0.5007, 0.4583, 0.4449, 0.4481, 0.4224,
    0.4659, 0.4353, 0.5259, 0.5270, 0.5916, 0.4420, 0.5951, 0.5110, 0.5741, 0.5384, 0.5744
]
fedavg_acc = [x * 100 for x in fedavg_acc]

# FedProx Logs
fedprox_loss = [
    2.2649, 1.7652, 1.5851, 1.2724, 1.1962, 0.9771, 0.9605, 0.9252, 0.9087, 0.9455,
    0.9288, 0.9451, 0.8073, 0.8192, 0.8768, 0.7807, 0.9149, 0.7826, 0.8022, 0.7945,
    0.7254, 0.7345, 0.7521, 0.6889, 0.7052, 0.7010, 0.7280, 0.7151, 0.6893, 0.7011,
    0.7505, 0.7947, 0.7017, 0.6725, 0.6878, 0.6626, 0.6459, 0.6243, 0.6515, 0.6370,
    0.6724, 0.7211, 0.7355, 0.7496, 0.6821, 0.6704, 0.6995, 0.7445, 0.7386, 0.6868, 0.7355
]

fedprox_acc = [
    0.0576, 0.3765, 0.3797, 0.5849, 0.5852, 0.6907, 0.6828, 0.7027, 0.6644, 0.6559,
    0.6600, 0.6656, 0.7273, 0.7135, 0.7094, 0.7311, 0.6676, 0.7340, 0.7159, 0.7287,
    0.7577, 0.7606, 0.7416, 0.7787, 0.7577, 0.7635, 0.7425, 0.7454, 0.7641, 0.7597,
    0.7243, 0.7220, 0.7530, 0.7673, 0.7515, 0.7705, 0.7702, 0.7951, 0.7711, 0.7691,
    0.7451, 0.7287, 0.7138, 0.7071, 0.7454, 0.7638, 0.7352, 0.7071, 0.7246, 0.7375, 0.7112
]
fedprox_acc = [x * 100 for x in fedprox_acc]

# Centralized Max Baseline Accuracy
CENTRALIZED_MAX_ACC = 96.52

# --- 2. Plotting Figures ---
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

# Plot 1: Test Accuracy vs Rounds
ax1.plot(rounds, fedavg_acc, label='FedAvg Global Accuracy', color='#d62728', linestyle='--', marker='o', markersize=3, alpha=0.85)
ax1.plot(rounds, fedprox_acc, label=r'FedProx Global Accuracy ($\mu=1.0$)', color='#2ca02c', linewidth=2.2, marker='s', markersize=3)
ax1.axhline(y=CENTRALIZED_MAX_ACC, color='#1f77b4', linestyle=':', linewidth=2, label=f'Centralized Baseline Peak ({CENTRALIZED_MAX_ACC}%)')

ax1.set_title('Global Model Test Accuracy over Federated Rounds', fontsize=13, fontweight='bold')
ax1.set_xlabel('Federated Communication Rounds', fontsize=11, fontweight='bold')
ax1.set_ylabel('Test Accuracy (%)', fontsize=11, fontweight='bold')
ax1.set_ylim([0, 105])
ax1.legend(loc='lower right', frameon=True, fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.6)

# Plot 2: Test Loss vs Rounds
ax2.plot(rounds, fedavg_loss, label='FedAvg Global Loss', color='#d62728', linestyle='--', marker='o', markersize=3, alpha=0.85)
ax2.plot(rounds, fedprox_loss, label=r'FedProx Global Loss ($\mu=1.0$)', color='#2ca02c', linewidth=2.2, marker='s', markersize=3)

ax2.set_title('Global Model Test Loss over Federated Rounds', fontsize=13, fontweight='bold')
ax2.set_xlabel('Federated Communication Rounds', fontsize=11, fontweight='bold')
ax2.set_ylabel('Cross-Entropy Loss', fontsize=11, fontweight='bold')
ax2.set_ylim([0, 2.5])
ax2.legend(loc='upper right', frameon=True, fontsize=10)
ax2.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.savefig("Federated_Convergence_Comparison.png", dpi=300, bbox_inches='tight')
print("✓ Success! Convergence curves saved as 'Federated_Convergence_Comparison.png'")
plt.show()
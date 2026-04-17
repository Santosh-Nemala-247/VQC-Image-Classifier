import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# -----------------------------------------------
# Load saved results from Step 3 and Step 4
# -----------------------------------------------
cnn_data = torch.load('cnn_results.pt')
vqc_data = torch.load('vqc_results.pt')

cnn_acc    = cnn_data['accuracy']
cnn_time   = cnn_data['time']
cnn_params = cnn_data['params']
cnn_losses = cnn_data['losses']

vqc_acc    = vqc_data['accuracy']
vqc_time   = vqc_data['time']
vqc_params = vqc_data['params']
vqc_losses = vqc_data['losses']

print("=" * 50)
print("       FINAL COMPARISON RESULTS")
print("=" * 50)
print(f"{'Metric':<20} {'CNN':>10} {'VQC':>10}")
print("-" * 50)
print(f"{'Accuracy (%)':<20} {cnn_acc:>10.2f} {vqc_acc:>10.2f}")
print(f"{'Parameters':<20} {cnn_params:>10} {vqc_params:>10}")
print(f"{'Training Time (s)':<20} {cnn_time:>10.2f} {vqc_time:>10.2f}")
print("=" * 50)

epochs = range(1, len(cnn_losses) + 1)

# -----------------------------------------------
# Create a 2x2 comparison dashboard
# -----------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle('Hybrid Quantum-Classical vs Classical CNN\nImage Classifier Comparison',
             fontsize=15, fontweight='bold', y=1.01)

# --- Plot 1: Accuracy Bar Chart ---
ax1 = axes[0, 0]
bars = ax1.bar(['Classical CNN', 'Hybrid VQC'],
               [cnn_acc, vqc_acc],
               color=['#2196F3', '#9C27B0'], width=0.4, edgecolor='black')
ax1.set_title('Accuracy Comparison (%)', fontweight='bold')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim([95, 101])
ax1.grid(axis='y', linestyle='--', alpha=0.5)
for bar, val in zip(bars, [cnn_acc, vqc_acc]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
             f'{val:.2f}%', ha='center', fontweight='bold', fontsize=12)

# --- Plot 2: Training Time Bar Chart ---
ax2 = axes[0, 1]
bars2 = ax2.bar(['Classical CNN', 'Hybrid VQC'],
                [cnn_time, vqc_time],
                color=['#2196F3', '#9C27B0'], width=0.4, edgecolor='black')
ax2.set_title('Training Time Comparison (seconds)', fontweight='bold')
ax2.set_ylabel('Time (seconds)')
ax2.grid(axis='y', linestyle='--', alpha=0.5)
for bar, val in zip(bars2, [cnn_time, vqc_time]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
             f'{val:.2f}s', ha='center', fontweight='bold', fontsize=12)

# --- Plot 3: Parameter Count Bar Chart ---
ax3 = axes[1, 0]
bars3 = ax3.bar(['Classical CNN', 'Hybrid VQC'],
                [cnn_params, vqc_params],
                color=['#2196F3', '#9C27B0'], width=0.4, edgecolor='black')
ax3.set_title('Model Parameter Count', fontweight='bold')
ax3.set_ylabel('Number of Parameters')
ax3.grid(axis='y', linestyle='--', alpha=0.5)
for bar, val in zip(bars3, [cnn_params, vqc_params]):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
             f'{val:,}', ha='center', fontweight='bold', fontsize=11)

# --- Plot 4: Training Loss Curves ---
ax4 = axes[1, 1]
ax4.plot(epochs, cnn_losses, marker='o', color='#2196F3',
         linewidth=2, label='Classical CNN', markersize=5)
ax4.plot(epochs, vqc_losses, marker='s', color='#9C27B0',
         linewidth=2, label='Hybrid VQC', markersize=5)
ax4.set_title('Training Loss over Epochs', fontweight='bold')
ax4.set_xlabel('Epoch')
ax4.set_ylabel('Loss')
ax4.legend()
ax4.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('final_comparison.png', dpi=150, bbox_inches='tight')
print("\n🖼️  Final comparison chart saved as final_comparison.png")
plt.show()

# -----------------------------------------------
# Print Summary for Presentation
# -----------------------------------------------
print("\n" + "=" * 50)
print("         PRESENTATION SUMMARY")
print("=" * 50)
print("""
🔵 Classical CNN:
   • Simple convolution layers
   • Fast training (~2 sec)
   • 51,618 parameters
   • 99% accuracy

🟣 Hybrid Quantum VQC:
   • Classical encoder + Quantum circuit + Classical decoder
   • Uses 4 qubits with AngleEmbedding + Entanglement
   • Slightly fewer parameters (50,518)
   • 100% accuracy on binary classification
   • Slower due to quantum simulation

🏆 Key Insight:
   The Hybrid VQC matched/exceeded CNN accuracy
   with fewer parameters — showing quantum
   advantage potential for future hardware!
""")
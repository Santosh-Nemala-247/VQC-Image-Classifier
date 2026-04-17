import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import pennylane as qml
import numpy as np
import matplotlib.pyplot as plt
import time

# -----------------------------------------------
# Reload Data (same as before)
# -----------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = torchvision.datasets.MNIST(root='./data', train=True,  download=False, transform=transform)
test_data  = torchvision.datasets.MNIST(root='./data', train=False, download=False, transform=transform)

def filter_01(dataset, max_samples=400):
    indices = [i for i, (img, label) in enumerate(dataset) if label in [0, 1]]
    indices = indices[:max_samples]
    imgs   = torch.stack([dataset[i][0] for i in indices])
    labels = torch.tensor([dataset[i][1] for i in indices])
    return imgs, labels

X_train, y_train = filter_01(train_data, 400)
X_test,  y_test  = filter_01(test_data,  100)

train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset  = torch.utils.data.TensorDataset(X_test,  y_test)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=32, shuffle=False)

print("✅ Data loaded!")

# -----------------------------------------------
# QUANTUM PART: Define the quantum circuit
# -----------------------------------------------
# We use 4 qubits — each qubit is like a quantum "bit"
N_QUBITS = 4
N_LAYERS = 2  # How many layers of quantum gates

# Create a quantum device (simulator running on your CPU)
dev = qml.device("default.qubit", wires=N_QUBITS)

# Define the quantum circuit
@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    # Step 1: Encode classical data into quantum states
    # AngleEmbedding rotates each qubit based on input values
    qml.AngleEmbedding(inputs, wires=range(N_QUBITS))

    # Step 2: Apply trainable quantum gates (this is what gets learned!)
    # BasicEntanglerLayers creates entanglement between qubits
    qml.BasicEntanglerLayers(weights, wires=range(N_QUBITS))

    # Step 3: Measure qubits — gives us output values between -1 and 1
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]

# -----------------------------------------------
# HYBRID MODEL: Classical + Quantum layers
# -----------------------------------------------
class HybridVQC(nn.Module):
    def __init__(self):
        super(HybridVQC, self).__init__()

        # Classical pre-processing: shrinks 784 pixels → 4 numbers for qubits
        self.classical_pre = nn.Sequential(
            nn.Linear(28 * 28, 64),   # 784 → 64
            nn.ReLU(),
            nn.Linear(64, N_QUBITS),  # 64 → 4 (one per qubit)
            nn.Tanh()                 # Keep values between -1 and 1
        )

        # Quantum layer weights (these get trained like normal weights)
        weight_shapes = {"weights": (N_LAYERS, N_QUBITS)}
        self.quantum_layer = qml.qnn.TorchLayer(quantum_circuit, weight_shapes)

        # Classical post-processing: quantum output → final prediction
        self.classical_post = nn.Linear(N_QUBITS, 2)  # 4 → 2 classes

    def forward(self, x):
        x = x.view(-1, 28 * 28)          # Flatten image: [batch, 1, 28, 28] → [batch, 784]
        x = self.classical_pre(x)         # Classical: 784 → 4
        x = self.quantum_layer(x)         # Quantum circuit processes 4 values
        x = self.classical_post(x)        # Classical: 4 → 2 (final output)
        return x

# Create model
vqc_model = HybridVQC()
vqc_params = sum(p.numel() for p in vqc_model.parameters())
print("✅ Hybrid VQC Model created!")
print(f"📊 Total parameters: {vqc_params}")

# -----------------------------------------------
# Train the VQC Model
# -----------------------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(vqc_model.parameters(), lr=0.01)

EPOCHS = 10
train_losses = []

print("\n⚛️  Starting Hybrid VQC Training...")
print("   (This is slower than CNN — quantum simulation takes more time!)\n")
start_time = time.time()

for epoch in range(EPOCHS):
    vqc_model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = vqc_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f}")

vqc_time = time.time() - start_time
print(f"\n⏱️  VQC Training Time: {vqc_time:.2f} seconds")

# -----------------------------------------------
# Test the VQC Model
# -----------------------------------------------
vqc_model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = vqc_model(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

vqc_accuracy = correct / total * 100

print(f"\n📊 VQC RESULTS:")
print(f"   ✅ Accuracy      : {vqc_accuracy:.2f}%")
print(f"   ✅ Parameters    : {vqc_params}")
print(f"   ✅ Training Time : {vqc_time:.2f} seconds")

# Save results
torch.save({
    'accuracy': vqc_accuracy,
    'time': vqc_time,
    'params': vqc_params,
    'losses': train_losses
}, 'vqc_results.pt')
print("\n💾 Results saved to vqc_results.pt")

# Plot loss
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', color='purple')
plt.title('VQC Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('vqc_loss.png')
print("📈 Loss graph saved as vqc_loss.png")
plt.show()
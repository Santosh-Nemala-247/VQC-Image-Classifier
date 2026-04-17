import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import time

# -----------------------------------------------
# Reload Data (same as Step 2)
# -----------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = torchvision.datasets.MNIST(root='./data', train=True,  download=True, transform=transform)
test_data  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

def filter_01(dataset, max_samples=400):
    indices = [i for i, (img, label) in enumerate(dataset) if label in [0, 1]]
    indices = indices[:max_samples]
    imgs   = torch.stack([dataset[i][0] for i in indices])
    labels = torch.tensor([dataset[i][1] for i in indices])
    return imgs, labels

X_train, y_train = filter_01(train_data, 400)
X_test,  y_test  = filter_01(test_data,  100)

# Convert to DataLoader (batches)
train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
test_dataset  = torch.utils.data.TensorDataset(X_test,  y_test)
train_loader  = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader   = torch.utils.data.DataLoader(test_dataset,  batch_size=32, shuffle=False)

print("✅ Data loaded!")

# -----------------------------------------------
# Define Classical CNN Model
# -----------------------------------------------
# CNN = Convolutional Neural Network
# It looks at small patches of the image to find patterns
class ClassicalCNN(nn.Module):
    def __init__(self):
        super(ClassicalCNN, self).__init__()

        # Layer 1: Finds edges and basic shapes
        self.conv1 = nn.Conv2d(1, 8, kernel_size=3, padding=1)  # 1 channel in, 8 filters out

        # Layer 2: Finds more complex patterns
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1) # 8 in, 16 filters out

        # Pooling: Shrinks image size (28x28 → 14x14 → 7x7)
        self.pool = nn.MaxPool2d(2, 2)

        # Activation: Adds non-linearity (makes model smarter)
        self.relu = nn.ReLU()

        # Fully connected layers: Makes final decision
        self.fc1 = nn.Linear(16 * 7 * 7, 64)  # 784 inputs → 64 neurons
        self.fc2 = nn.Linear(64, 2)            # 64 → 2 outputs (digit 0 or 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Conv1 → ReLU → Pool
        x = self.pool(self.relu(self.conv2(x)))  # Conv2 → ReLU → Pool
        x = x.view(-1, 16 * 7 * 7)              # Flatten to 1D
        x = self.relu(self.fc1(x))              # Fully connected 1
        x = self.fc2(x)                         # Fully connected 2 (output)
        return x

# Create model
cnn_model = ClassicalCNN()
print("✅ CNN Model created!")
print(f"📊 Total parameters: {sum(p.numel() for p in cnn_model.parameters())}")

# -----------------------------------------------
# Train the CNN
# -----------------------------------------------
criterion = nn.CrossEntropyLoss()           # Measures how wrong predictions are
optimizer = optim.Adam(cnn_model.parameters(), lr=0.001)  # Updates weights smartly

EPOCHS = 10  # Number of times we go through all training data
train_losses = []

print("\n🚀 Starting CNN Training...")
start_time = time.time()

for epoch in range(EPOCHS):
    cnn_model.train()
    total_loss = 0

    for images, labels in train_loader:
        optimizer.zero_grad()           # Clear old gradients
        outputs = cnn_model(images)     # Forward pass
        loss = criterion(outputs, labels)  # Calculate loss
        loss.backward()                 # Backpropagation
        optimizer.step()               # Update weights
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    train_losses.append(avg_loss)
    print(f"  Epoch {epoch+1:2d}/{EPOCHS} | Loss: {avg_loss:.4f}")

cnn_time = time.time() - start_time
print(f"\n⏱️  CNN Training Time: {cnn_time:.2f} seconds")

# -----------------------------------------------
# Test the CNN
# -----------------------------------------------
cnn_model.eval()
correct = 0
total   = 0

with torch.no_grad():
    for images, labels in test_loader:
        outputs = cnn_model(images)
        _, predicted = torch.max(outputs, 1)
        total   += labels.size(0)
        correct += (predicted == labels).sum().item()

cnn_accuracy = correct / total * 100
cnn_params   = sum(p.numel() for p in cnn_model.parameters())

print(f"\n📊 CNN RESULTS:")
print(f"   ✅ Accuracy      : {cnn_accuracy:.2f}%")
print(f"   ✅ Parameters    : {cnn_params}")
print(f"   ✅ Training Time : {cnn_time:.2f} seconds")

# Save results for later comparison
torch.save({
    'accuracy': cnn_accuracy,
    'time': cnn_time,
    'params': cnn_params,
    'losses': train_losses
}, 'cnn_results.pt')
print("\n💾 Results saved to cnn_results.pt")

# Plot training loss
plt.figure(figsize=(8, 4))
plt.plot(range(1, EPOCHS+1), train_losses, marker='o', color='blue')
plt.title('CNN Training Loss over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.tight_layout()
plt.savefig('cnn_loss.png')
print("📈 Loss graph saved as cnn_loss.png")
plt.show()
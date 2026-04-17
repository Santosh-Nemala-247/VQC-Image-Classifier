import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# -----------------------------------------------
# STEP 2: Load MNIST and filter only 0s and 1s
# -----------------------------------------------

# This converts images to tensors and normalizes pixel values to [-1, 1]
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Download MNIST dataset (downloads automatically first time)
print("📥 Downloading MNIST dataset...")
train_data = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_data  = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# -----------------------------------------------
# Filter: Keep only images labeled 0 or 1
# -----------------------------------------------
def filter_01(dataset):
    # Get indices where label is 0 or 1
    indices = [i for i, (img, label) in enumerate(dataset) if label in [0, 1]]
    # Pick only first 400 for speed
    indices = indices[:400]
    imgs   = torch.stack([dataset[i][0] for i in indices])
    labels = torch.tensor([dataset[i][1] for i in indices])
    return imgs, labels

print("🔍 Filtering digits 0 and 1 only...")
X_train, y_train = filter_01(train_data)
X_test,  y_test  = filter_01(test_data)

print(f"✅ Training samples : {len(X_train)}")
print(f"✅ Testing  samples : {len(X_test)}")
print(f"✅ Image shape      : {X_train[0].shape}")  # Should be [1, 28, 28]

# -----------------------------------------------
# Show a few sample images
# -----------------------------------------------
fig, axes = plt.subplots(1, 6, figsize=(10, 2))
fig.suptitle("Sample MNIST Images (0s and 1s)", fontsize=13)
for i, ax in enumerate(axes):
    ax.imshow(X_train[i].squeeze(), cmap='gray')
    ax.set_title(f"Label: {y_train[i].item()}")
    ax.axis('off')
plt.tight_layout()
plt.savefig("sample_images.png")
print("🖼️  Sample image saved as sample_images.png")
plt.show()
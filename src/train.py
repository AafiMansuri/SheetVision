# Aafi Mansuri, Terry Zhen
# Training for CNN Model Using Datasets
# Outputs saved model for Phase 4 predictions

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_architecture import MusicCNN
import matplotlib.pyplot as plt

# Helper function to plot accuracy
def plot_accuracy(train_accs, val_accs, save_path="accuracy_plot.png"):
    epochs = range(1, len(train_accs) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, marker='o', label='Train Accuracy')
    plt.plot(epochs, val_accs, marker='o', label='Val Accuracy')
    plt.title('Train vs Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xticks(epochs)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")

# Plot train and val losses
def plot_loss(training_losses, validation_losses, save_path="loss_plot.png"):
    epochs = range(1, len(training_losses) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, training_losses, marker='o', label='Train Loss')
    plt.plot(epochs, validation_losses,   marker='o', label='Val Loss')
    plt.title('Train vs Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Plot saved to {save_path}")

# Path to dataset root
DATASET_PATH = "../dataset"

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

# Load dataset and infer class labels from subfolder names
dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
print(dataset.class_to_idx)

n = len(dataset)
print("Length of dataset: " + str(n))

# 80/20 train/val split
n_val = int(n * 0.20)
train_set, val_set = torch.utils.data.random_split(dataset, [n - n_val, n_val])

# Train and val dataloaders, shuffled train loader to reduce order bias
train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set,   batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize model
model = MusicCNN().to(device)
# Adam optimizer with learning rate of 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Loss eval
criterion = nn.CrossEntropyLoss()

train_accuracies, val_accuracies = [], []
train_losses, val_losses = [], []

# Train CNN model
for epoch in range(10):
    # Training pass
    model.train()
    train_correct, train_total = 0, 0
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        # Track training accuracy and loss
        train_correct += (preds.argmax(1) == labels).sum().item()
        train_total   += labels.size(0)
        running_loss += loss.item() * imgs.size(0)

    train_acc = train_correct / train_total
    train_accuracies.append(train_acc)
    train_losses.append(running_loss / train_total)

    # Validation pass
    model.eval()
    val_correct, val_total = 0, 0
    val_running_loss = 0.0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            # Track validation accuracy and loss
            val_preds = model(imgs)
            val_correct += (val_preds.argmax(1) == labels).sum().item()
            val_total   += labels.size(0)
            val_running_loss += criterion(val_preds, labels).item() * imgs.size(0)

    val_acc = val_correct / val_total
    val_accuracies.append(val_acc)
    val_losses.append(val_running_loss / val_total)

    print(f"Epoch {epoch+1:02d}  train_acc={train_acc:.3f}  val_acc={val_acc:.3f}")

# Save trained model
torch.save(model.state_dict(), "music_cnn.pt")

# Plot accuracy curves
plot_accuracy(train_accuracies, val_accuracies)

#Plot loss curves
plot_loss(train_losses, val_losses)
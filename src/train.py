# Aafi Mansuri, Terry Zhen
# Training for CNN Model Using Datasets
# Outputs saved model for Phase 4 predictions

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cnn_architecture import MusicCNN

DATASET_PATH = "../dataset"


transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),
])

dataset = datasets.ImageFolder(DATASET_PATH, transform=transform)
print(dataset.class_to_idx)

n = len(dataset)
print("Length of dataset: " + str(n))

n_val = int(n * 0.20)
train_set, val_set = torch.utils.data.random_split(dataset, [n - n_val, n_val])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
val_loader = DataLoader(val_set,   batch_size=32)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize model
model = MusicCNN().to(device)
# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
# Loss eval
criterion = nn.CrossEntropyLoss()

# Train CNN model
for epoch in range(10):
    model.train()
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            correct += (model(imgs).argmax(1) == labels).sum().item()
            total   += labels.size(0)

    print(f"Epoch {epoch+1:02d}  val_acc={correct/total:.3f}")

# Save trained model
torch.save(model.state_dict(), "music_cnn.pt")
import os
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# =========================
# CONFIG
# =========================
base_dir = os.path.expanduser('~/hack/diseases/cattle_diseases_data')
train_csv = os.path.join(base_dir, 'train/_classes.csv')
valid_csv = os.path.join(base_dir, 'valid/_classes.csv')
test_csv  = os.path.join(base_dir, 'test/_classes.csv')

train_img_dir = os.path.join(base_dir, 'train')
valid_img_dir = os.path.join(base_dir, 'valid')
test_img_dir  = os.path.join(base_dir, 'test')

batch_size = 16
num_epochs = 10
learning_rate = 1e-4
img_size = 224
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =========================
# DATASET
# =========================
class RoboflowDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.labels = self.df.columns.tolist()[1:]  # all columns except 'image'

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['filename'])
        img_path = os.path.expanduser(img_path)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = Image.open(img_path).convert('RGB')
        label = row[self.labels].values.astype(float)
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.float32)

# =========================
# TRANSFORMS
# =========================
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
])

# =========================
# DATA LOADERS
# =========================
train_dataset = RoboflowDataset(train_csv, train_img_dir, transform)
valid_dataset = RoboflowDataset(valid_csv, valid_img_dir, transform)
test_dataset = RoboflowDataset(test_csv, test_img_dir, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

num_labels = len(train_dataset.labels)
print(f"Number of classes: {num_labels}")

# =========================
# MODEL
# =========================
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_labels)
model = model.to(device)

# =========================
# LOSS & OPTIMIZER
# =========================
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# =========================
# TRAINING LOOP
# =========================
best_val_loss = float('inf')

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
    val_loss /= len(valid_loader.dataset)

    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}")

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(base_dir, 'best_model.pth'))
        print("Saved best model.")

# =========================
# TESTING
# =========================
model.eval()
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item() * images.size(0)
test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")

print(f"Training complete. Model saved at {os.path.join(base_dir, 'best_model.pth')}")

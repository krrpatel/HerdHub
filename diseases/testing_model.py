import torch
from torchvision import models, transforms
from PIL import Image

# ---- Device and Model ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 11

model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load trained model
state_dict = torch.load('best_model.pth', map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# ---- Transform ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ---- Hardcoded label names ----
label_cols = ['BRD', 'Bovine', 'Contagious', 'Dermatitis', 'Disease', 
              'Ecthym', 'Respiratory', 'Unlabeled', 'healthy', 'lumpy', 'skin']

# ---- User input for image path ----
img_path = input("Enter the full path of the image: ").strip()

try:
    image = Image.open(img_path).convert("RGB")
except Exception as e:
    print("Error opening image:", e)
    exit()

# ---- Predict ----
input_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.sigmoid(outputs)[0]  # probabilities for each label

# Get predicted labels with confidence
predicted_labels = [(label_cols[i], float(probs[i])) for i in range(num_classes) if probs[i] > 0.5]

# ---- Print results ----
print(f"\nFilename: {img_path}")
if predicted_labels:
    for label, conf in predicted_labels:
        print(f"{label}: {conf*100:.2f}%")
else:
    print("No label predicted above 50% confidence")

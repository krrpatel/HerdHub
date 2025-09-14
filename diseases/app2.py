import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify
import google.generativeai as genai
import os
import requests
from io import BytesIO

# ------------------- Flask App -------------------
app = Flask(__name__)

# ------------------- Model Setup -------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 11

# Initialize ResNet50
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)

# Load trained weights
state_dict = torch.load('best_model.pth', map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Hardcoded label names
label_cols = [
    'BRD', 'Bovine', 'Contagious', 'Dermatitis', 'Disease',
    'Ecthym', 'Respiratory', 'Unlabeled', 'healthy', 'lumpy', 'skin'
]

# ------------------- Gemini Setup -------------------
genai.configure(api_key=os.getenv("api"))  # set environment variable `api` with your key
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def get_gemini_info(disease_name):
    """Query Gemini 2.0 Flash for symptoms & handling steps"""
    if disease_name.lower() in ["healthy", "unlabeled"]:
        return "No disease detected. Cattle appears healthy."
    prompt = f"Give symptoms and steps to handle cattle disease: {disease_name}"
    response = gemini_model.generate_content(prompt)
    return response.text if response else "No info available."

# ------------------- Prediction Utility -------------------
def predict_image(image: Image.Image):
    """Run model prediction on PIL image"""
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)[0]

    predictions = [
        {"label": label_cols[i], "confidence": float(probs[i]) * 100}
        for i in range(num_classes) if probs[i] > 0.5
    ]

    top_prediction = max(predictions, key=lambda x: x["confidence"], default=None)
    gemini_info = get_gemini_info(top_prediction["label"]) if top_prediction else "No prediction above 50%."
    return predictions, gemini_info

# ------------------- API Route -------------------
@app.route("/predict", methods=["POST"])
def predict():
    # 1️⃣ Check for file upload
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        try:
            image = Image.open(file).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image: {e}"}), 400

    # 2️⃣ Check for JSON URL
    elif request.is_json:
        data = request.get_json()
        if "url" not in data:
            return jsonify({"error": "No URL provided"}), 400
        try:
            response = requests.get(data["url"])
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image URL: {e}"}), 400

    else:
        return jsonify({"error": "No file or URL provided"}), 400

    # Run prediction
    predictions, gemini_info = predict_image(image)

    filename_or_url = file.filename if "file" in request.files else data["url"]

    return jsonify({
        "filename_or_url": filename_or_url,
        "predictions": predictions,
        "gemini_info": gemini_info
    })

# ------------------- Run Server -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)

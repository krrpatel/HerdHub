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

# Hardcoded label names (only used internally for Gemini)
label_cols = [
    'BRD', 'Bovine', 'Contagious', 'Dermatitis', 'Disease',
    'Ecthym', 'Respiratory', 'Unlabeled', 'healthy', 'lumpy', 'skin'
]

# ------------------- Gemini Setup -------------------
genai.configure(api_key=os.getenv("api"))  # set environment variable `api` with your key
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

def get_gemini_info(disease_name, language="en"):
    """Query Gemini for disease name & handling steps in English or Hindi"""

    if disease_name.lower() == "healthy":
        if language == "hi":
            return {
                "disease_name": "स्वस्थ",
                "disease_info": "कोई बीमारी नहीं पाई गई। पशु स्वस्थ दिखाई दे रहा है।"
            }
        else:
            return {
                "disease_name": "Healthy",
                "disease_info": "No disease detected. Cattle appears healthy."
            }

    if disease_name.lower() == "unlabeled":
        if language == "hi":
            return {
                "disease_name": "गलत छवि",
                "disease_info": "गलत छवि या छवि मेल नहीं खाती।"
            }
        else:
            return {
                "disease_name": "Wrong image",
                "disease_info": "Wrong image or image mismatch."
            }

    # If not healthy or unlabeled → ask Gemini
    if language == "hi":
        prompt_name = f"Write the cattle disease name '{disease_name}' in Hindi (one or two words only, strictly no extra question only one or two words) In Hindi."
        prompt_info = f"Give symptoms and steps to handle cattle disease '{disease_name}' in Hindi."
    else:
        prompt_name = f"Give only the exact disease name for: {disease_name} In Strictly One Or Two Words"
        prompt_info = f"Give symptoms and steps to handle cattle disease: {disease_name}"

    # Ask Gemini for name
    response_name = gemini_model.generate_content(prompt_name)
    name_text = response_name.text.strip() if response_name else disease_name

    # Ask Gemini for info
    response_info = gemini_model.generate_content(prompt_info)
    info_text = response_info.text.strip() if response_info else "No info available."

    return {
        "disease_name": name_text,
        "disease_info": info_text
    }
# ------------------- Prediction Utility -------------------
def predict_image(image: Image.Image, language="en"):
    """Run model prediction on PIL image and return confidence + Gemini info"""
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.sigmoid(outputs)[0]

    # Find top prediction
    top_index = torch.argmax(probs).item()
    confidence = float(probs[top_index]) * 100

    # Get Gemini info for that label
    gemini_info = get_gemini_info(label_cols[top_index], language)

    return confidence, gemini_info

# ------------------- API Route -------------------
@app.route("/predict", methods=["POST"])
def predict():
    language = "en"  # default
    data = None

    if request.is_json:
        data = request.get_json()
        language = data.get("language", "en")

    # 1️⃣ File upload
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == "":
            return jsonify({"error": "Empty filename"}), 400
        try:
            image = Image.open(file).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image: {e}"}), 400

    # 2️⃣ URL upload
    elif data and "url" in data:
        try:
            response = requests.get(data["url"])
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except Exception as e:
            return jsonify({"error": f"Invalid image URL: {e}"}), 400
    else:
        return jsonify({"error": "No file or URL provided"}), 400

    # Run prediction
    confidence, gemini_output = predict_image(image, language)

    filename_or_url = file.filename if "file" in request.files else data["url"]

    return jsonify({
        "filename_or_url": filename_or_url,
        "confidence": confidence,
        "language": language,
        "disease_name": gemini_output["disease_name"],
        "disease_info": gemini_output["disease_info"]
    })

# ------------------- Run Server -------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=6000, debug=True)

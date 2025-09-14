from flask import Flask, request, jsonify
import joblib
import numpy as np

# -----------------------------
# Load Models and Encoders
# -----------------------------
trained_models = joblib.load("milk_yield_models.pkl")
le_animal = joblib.load("animal_encoder.pkl")
le_breed = joblib.load("breed_encoder.pkl")
scaler = joblib.load("feature_scaler.pkl")

# Pick best model (highest R² when saved)
best_model_name = max(trained_models.keys(), key=lambda x: hasattr(trained_models[x], "feature_importances_"))

# Flask app
app = Flask(__name__)

# -----------------------------
# Prediction Function
# -----------------------------
def predict_with_all_models(data):
    try:
        # Encode categorical inputs
        try:
            animal_enc = le_animal.transform([data["animal_type"]])[0]
        except:
            animal_enc = 0

        try:
            breed_enc = le_breed.transform([data["breed"]])[0]
        except:
            breed_enc = 0

        # Create input array
        input_features = np.array([
            animal_enc,
            breed_enc,
            float(data["age_years"]),
            int(data["lactation_number"]),
            float(data["days_in_milk"]),
            float(data["feed_given_kg"]),
            float(data["water_intake_liters"]),
            int(data["body_condition_score"]),
            float(data["temperature_c"]),
            float(data["humidity_percent"])
        ]).reshape(1, -1)

        # Scaled version
        input_features_scaled = scaler.transform(input_features)

        predictions = {}
        for name, model in trained_models.items():
            if name in ["SVR", "Ridge Regression", "Linear Regression"]:
                pred = model.predict(input_features_scaled)[0]
            else:
                pred = model.predict(input_features)[0]
            predictions[name] = max(0, float(pred))  # ensure non-negative

        avg_prediction = float(np.mean(list(predictions.values())))
        best_prediction = predictions[best_model_name]

        return {
            "predictions": predictions,
            "average_prediction": avg_prediction,
            "best_model": best_model_name,
            "best_model_prediction": best_prediction
        }

    except Exception as e:
        return {"error": str(e)}

# -----------------------------
# API Route
# -----------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    result = predict_with_all_models(data)
    return jsonify(result)

# -----------------------------
# Run App
# -----------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

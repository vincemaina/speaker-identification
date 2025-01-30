from flask import Flask, request, jsonify
import joblib
import requests
import os
import io

MODEL_URL = "https://your-bucket-name.s3.amazonaws.com/model.pkl"

def load_model():
    """Downloads the model from external storage & loads it into memory."""
    response = requests.get(MODEL_URL)
    model = joblib.load(io.BytesIO(response.content))  # Load model in-memory
    return model

model = load_model()

def predict(request):
    """Handles HTTP requests for predictions."""
    data = request.get_json()
    features = data.get("features", [])
    prediction = model.predict([features])
    return jsonify({"prediction": prediction.tolist()})

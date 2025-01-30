import joblib
import functions_framework
import os
import numpy as np
from flask import jsonify, make_response

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
model = joblib.load(MODEL_PATH)


@functions_framework.http
def predict(request):
    """Receives JSON feature vector and returns prediction."""
    
    # Handle CORS preflight requests
    if request.method == "OPTIONS":
        response = make_response()
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    try:
        # Parse JSON input
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)  # Ensure 2D array

        # Make prediction using the ML model
        prediction = model.predict(features)
        
        print("Prediction:", prediction)

        response = jsonify({"prediction": prediction.tolist()})
        response.headers["Access-Control-Allow-Origin"] = "*"  # Allow frontend requests
        return response

    except Exception as e:
        response = jsonify({"error": str(e)})
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response, 500

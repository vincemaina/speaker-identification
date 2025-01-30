import joblib
from flask import Flask
import requests
import os
import io

app = Flask(__name__)

MODEL_URL = "https://speaker-identification.vercel.app/svm.pkl"
MODEL_PATH = "model.pkl"

def download_model():
    """Downloads model.pkl from an external URL if not already downloaded."""
    if not os.path.exists(MODEL_PATH):
        print("Downloading model...")
        response = requests.get(MODEL_URL)
        with open(MODEL_PATH, "wb") as file:
            file.write(response.content)
        print("Download complete.")
    else:
        print("Model already downloaded.")

@app.route("/api/python")
def hello_world():
    print("Reached here")
    
    response = requests.get(MODEL_URL)
    model = joblib.load(io.BytesIO(response.content))
    
    print(type(model))
    return f"Model: {model}"

if __name__ == "__main__":
    app.run(debug=True)

import joblib
from pathlib import Path
from flask import Flask
app = Flask(__name__)

# Ensure we get the correct path
MODEL_PATH = Path(__file__).parent / "svm.pkl"
model = joblib.load(MODEL_PATH)  # Load model with preprocessing

@app.route("/api/python")
def hello_world():
    return "<p>Hello, World!</p>"
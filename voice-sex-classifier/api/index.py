import joblib
from pathlib import Path
from flask import Flask
app = Flask(__name__)

# Ensure we get the correct path
model = joblib.load("https://speaker-identification.vercel.app/svm.plk")  # Load model with preprocessing

@app.route("/api/python")
def hello_world():
    return "<p>Hello, World!</p>"
import joblib
from flask import Flask
app = Flask(__name__)

model = joblib.load("svm.pkl")  # Load model with preprocessing

@app.route("/api/python")
def hello_world():
    return "<p>Hello, World!</p>"
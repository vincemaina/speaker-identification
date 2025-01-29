import joblib
from urllib.request import urlopen
from flask import Flask
app = Flask(__name__)

model = joblib.load(urlopen("https://speaker-identification.vercel.app/svm.plk"))

@app.route("/api/python")
def hello_world():
    return "<p>Hello, World!</p>"
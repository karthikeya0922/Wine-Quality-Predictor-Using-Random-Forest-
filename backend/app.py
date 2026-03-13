import os
import pickle

import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

app = Flask(__name__)

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Maps JSON field names → DataFrame column names expected by the model
COLUMN_MAP = {
    "fixed_acidity":       "fixed acidity",
    "volatile_acidity":    "volatile acidity",
    "citric_acid":         "citric acid",
    "residual_sugar":      "residual sugar",
    "chlorides":           "chlorides",
    "free_sulfur_dioxide": "free sulfur dioxide",
    "total_sulfur_dioxide":"total sulfur dioxide",
    "density":             "density",
    "ph":                  "pH",
    "sulphates":           "sulphates",
    "alcohol":             "alcohol",
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    missing = [k for k in COLUMN_MAP if k not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        wine = pd.DataFrame([{col: float(data[key]) for key, col in COLUMN_MAP.items()}])
    except (TypeError, ValueError):
        return jsonify({"error": "All field values must be numeric"}), 400

    quality = int(model.predict(wine)[0])
    return jsonify({"quality": quality})


if __name__ == "__main__":
    app.run(debug=True)

from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

# Create Flask app
app = Flask(__name__)

# Load the saved model
model = pickle.load(open("model.pkl", "rb"))

# Route 1 - Show the webpage
@app.route("/")
def home():
    return render_template("index.html")

# Route 2 - Predict wine quality
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json  # Get data sent from browser

    # Create a DataFrame with the input values
    new_wine = pd.DataFrame({
        "fixed acidity":        [data["fixed_acidity"]],
        "volatile acidity":     [data["volatile_acidity"]],
        "citric acid":          [data["citric_acid"]],
        "residual sugar":       [data["residual_sugar"]],
        "chlorides":            [data["chlorides"]],
        "free sulfur dioxide":  [data["free_sulfur_dioxide"]],
        "total sulfur dioxide": [data["total_sulfur_dioxide"]],
        "density":              [data["density"]],
        "pH":                   [data["ph"]],
        "sulphates":            [data["sulphates"]],
        "alcohol":              [data["alcohol"]]
    })

    prediction = model.predict(new_wine)
    quality = int(prediction[0])

    return jsonify({"quality": quality})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)

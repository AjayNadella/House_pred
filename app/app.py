from flask import Flask, request, jsonify
import joblib
import numpy as np

print("Starting the Flask application...")

app = Flask(__name__)

print("Loading the model...")
model = joblib.load('../models/model.pkl')
print("Model loaded successfully.")

@app.route('/')
def home():
    return "Welcome to the Housing Price Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    print("Received a request for prediction.")
    data = request.get_json()  # Get data posted as a JSON
    features = data['features']
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    print(f"Prediction: {prediction[0]}")
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    print("Running the Flask app...")
    app.run(debug=True)

import os
import joblib
import json
import logging
import pandas as pd
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

# Set up logging
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
logging.basicConfig(filename=os.path.join(LOG_DIR, "api.log"), level=logging.INFO, 
                    format="%(asctime)s - %(levelname)s - %(message)s")

# Load the trained model
MODEL_PATH = "saved_models/random_forest.pkl"  # Adjust based on the best model
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    logging.info(f"Model loaded from {MODEL_PATH}")
else:
    logging.error("Model file not found!")
    model = None

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Fraud Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    """Predict fraud probability based on incoming JSON data."""
    try:
        data = request.get_json()
        df = pd.DataFrame([data])  # Convert JSON to DataFrame
        
        if model is None:
            logging.error("Prediction attempted without a loaded model.")
            return jsonify({"error": "Model not loaded"}), 500

        prediction = model.predict(df)[0]
        response = {"fraud_prediction": int(prediction)}
        
        logging.info(f"Prediction made: {json.dumps(data)} → {response}")
        return jsonify(response)

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

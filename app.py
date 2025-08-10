from flask import Flask, request, render_template, Response
import mlflow.sklearn
import numpy as np
import joblib
import mlflow.pyfunc
import logging
from logging.handlers import RotatingFileHandler
import os
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import time

app = Flask(__name__, template_folder='templates')
# ---------------------- Logging Setup ----------------------
if not os.path.exists('logs'):
    os.makedirs('logs')

log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] - %(message)s')
file_handler = RotatingFileHandler('logs/app.log', maxBytes=1000000, backupCount=3)
file_handler.setFormatter(log_formatter)
file_handler.setLevel(logging.INFO)

app.logger.setLevel(logging.INFO)
app.logger.addHandler(file_handler)

# Define the path to your model inside the Docker container
MODEL_PATH = "mlflow_model"

# Load model and scaler
try:
    model = mlflow.sklearn.load_model(MODEL_PATH)
    scaler = joblib.load("scaler.pkl")
    app.logger.info("Model and scaler loaded successfully.")
except Exception as e:
    app.logger.exception("Failed to load model or scaler.")
    raise

label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests")
REQUEST_LATENCY = Histogram("predict_request_latency_seconds", "Prediction latency in seconds")

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    probabilities = None

    if request.method == "POST":
        start_time = time.time()
        try:
            REQUEST_COUNT.inc()
            features = [float(request.form[f"feature{i}"]) for i in range(1, 5)]
            app.logger.info(f"Received input features: {features}")

            # Scale the input
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)

            prediction_index = model.predict(features_scaled)[0]
            prediction = label_map.get(prediction_index, f"Unknown class: {prediction_index}")
            app.logger.info(f"Prediction result: {prediction}")

            # Probabilities
            probas = model.predict_proba(features_scaled)[0]
            probabilities = {label_map[i]: f"{prob:.2%}" for i, prob in enumerate(probas)}
            app.logger.info(f"Prediction probabilities: {probabilities}")

        except Exception as e:
            prediction = f"Error: {e}"
            app.logger.exception("Error during prediction.")
        # Observe latency
        REQUEST_LATENCY.observe(time.time() - start_time)
    return render_template("form.html", prediction=prediction, probabilities=probabilities)

@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

#This is the main changing the code
if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')

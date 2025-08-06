# from flask import Flask, request, render_template, Response
# import mlflow.sklearn
# import numpy as np
# import joblib
# import time
# import logging
# from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
#
# # Initialize Flask app
# app = Flask(__name__, template_folder='templates')
#
# # Setup logging
# logging.basicConfig(level=logging.INFO)
#
# # Load model and scaler
# MODEL_PATH = "mlflow_model"
# model = mlflow.sklearn.load_model(MODEL_PATH)
# scaler = joblib.load("scaler.pkl")
#
# # Label mapping
# label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}
#
# # --- Prometheus Metrics ---
# REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests")
# REQUEST_LATENCY = Histogram("predict_request_latency_seconds", "Prediction latency in seconds")
#
# @app.route("/", methods=["GET", "POST"])
# def predict():
#     prediction = None
#     probabilities = None
#
#     if request.method == "POST":
#         start_time = time.time()
#         REQUEST_COUNT.inc()
#
#         try:
#             # Extract and scale features
#             features = [float(request.form[f"feature{i}"]) for i in range(1, 5)]
#             features_array = np.array(features).reshape(1, -1)
#             features_scaled = scaler.transform(features_array)
#
#             # Predict
#             prediction_index = model.predict(features_scaled)[0]
#             prediction = label_map.get(prediction_index, f"Unknown class: {prediction_index}")
#
#             # Predict probabilities
#             probas = model.predict_proba(features_scaled)[0]
#             probabilities = {label_map[i]: f"{prob:.2%}" for i, prob in enumerate(probas)}
#
#             logging.info(f"Prediction: {prediction}, Probabilities: {probabilities}")
#
#         except Exception as e:
#             prediction = f"Error: {e}"
#
#         # Observe latency
#         REQUEST_LATENCY.observe(time.time() - start_time)
#
#     return render_template("form.html", prediction=prediction, probabilities=probabilities)
#
# @app.route("/metrics")
# def metrics():
#     return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)
#
# if __name__ == "__main__":
#     app.run(debug=True, port=8000, host="0.0.0.0")
from flask import Flask, request, render_template, Response
import mlflow.sklearn
import numpy as np
import joblib
import time
import logging
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

# --- Logging Configuration ---
# Configure logging to write to a file named 'app.log'.
# It will capture messages of level INFO and above.
# The format includes a timestamp, log level, and the message.
logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize Flask app
app = Flask(__name__, template_folder='templates')

# Load model and scaler
try:
    MODEL_PATH = "mlflow_model"
    model = mlflow.sklearn.load_model(MODEL_PATH)
    scaler = joblib.load("scaler.pkl")
    logging.info("Model and scaler loaded successfully.")
except Exception as e:
    logging.critical(f"Failed to load model or scaler: {e}")
    # Depending on the desired behavior, you might want to exit the app
    # import sys
    # sys.exit(1)


# Label mapping
label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

# --- Prometheus Metrics ---
REQUEST_COUNT = Counter("predict_requests_total", "Total prediction requests")
REQUEST_LATENCY = Histogram("predict_request_latency_seconds", "Prediction latency in seconds")

@app.route("/", methods=["GET", "POST"])
def predict():
    """
    Handles the prediction request. It takes features from a form,
    scales them, makes a prediction, and logs the result.
    """
    prediction = None
    probabilities = None

    if request.method == "POST":
        start_time = time.time()
        REQUEST_COUNT.inc()

        try:
            # Extract and scale features
            features = [float(request.form[f"feature{i}"]) for i in range(1, 5)]
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)

            # Predict
            prediction_index = model.predict(features_scaled)[0]
            prediction = label_map.get(prediction_index, f"Unknown class: {prediction_index}")

            # Predict probabilities
            probas = model.predict_proba(features_scaled)[0]
            probabilities = {label_map[i]: f"{prob:.2%}" for i, prob in enumerate(probas)}

            # Log the successful prediction to the file
            logging.info(f"Prediction successful. Features: {features}, Prediction: {prediction}, Probabilities: {probabilities}")

        except Exception as e:
            prediction = f"Error: {e}"
            # Log the error to the file
            logging.error(f"An error occurred during prediction: {e}", exc_info=True)

        # Observe latency
        REQUEST_LATENCY.observe(time.time() - start_time)

    return render_template("form.html", prediction=prediction, probabilities=probabilities)

@app.route("/metrics")
def metrics():
    """Exposes Prometheus metrics."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    # Use app.run for development. For production, use a WSGI server like Gunicorn.
    app.run(debug=True, port=8000, host="0.0.0.0")

from flask import Flask, request, render_template
import mlflow.sklearn
import numpy as np
import joblib
import mlflow.pyfunc

app = Flask(__name__, template_folder='templates')

# Define the path to your model inside the Docker container
MODEL_PATH = "mlflow_model" # This should match the COPY destination in your Dockerfile

# Load model from MLflow registry
# mlflow.set_tracking_uri("sqlite:///mlflow.db")
# model_name = "LogisticRegression"
# model_version = "Production"
# # model = mlflow.sklearn.load_model(f"models:/{model_name}/{model_version}")
model = mlflow.sklearn.load_model(MODEL_PATH)



# Load the MLflow model when the Flask app starts
# try:
#     model = mlflow.pyfunc.load_model(MODEL_PATH)
#     print(f"MLflow model loaded successfully from {MODEL_PATH}")
# except Exception as e:
#     print(f"Error loading MLflow model: {e}")
#     model = None # Handle the case where the model fails to load


# Load scaler
scaler = joblib.load("scaler.pkl")

label_map = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

@app.route("/", methods=["GET", "POST"])
def predict():

    prediction = None
    probabilities = None

    if request.method == "POST":
        try:
            features = [float(request.form[f"feature{i}"]) for i in range(1, 5)]

            # Scale input using the same scaler from training
            features_array = np.array(features).reshape(1, -1)
            features_scaled = scaler.transform(features_array)

            prediction_index  = model.predict(features_scaled)[0]
            prediction = label_map.get(prediction_index, f"Unknown class: {prediction_index}")

            # Get prediction probabilities
            probas = model.predict_proba(features_scaled)[0]
            probabilities = {label_map[i]: f"{prob:.2%}" for i, prob in enumerate(probas)}

        except Exception as e:
            prediction = f"Error: {e}"
    return render_template("form.html", prediction=prediction, probabilities=probabilities)
    # return render_template("form.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, port=8000, host='0.0.0.0')

import mlflow
import mlflow.sklearn
import joblib
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# def load_and_preprocess():
#     iris = load_iris(as_frame=True)
#     df = iris.frame
#     X = df.drop(columns=["target"])
#     y = df["target"]
#
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
#
#     # Save scaler
#     joblib.dump(scaler, "scaler.pkl")
#
#     return train_test_split(X_scaled, y, test_size=0.2, random_state=42)
import pandas as pd


def load_and_preprocess():
    # Load from local CSV file
    df = pd.read_csv("data/iris_processed.csv")
    X = df.drop(columns=["target"])
    y = df["target"]
    # Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Save scaler for use in the Flask app
    joblib.dump(scaler, "scaler.pkl")
    return train_test_split(X_scaled, y, test_size=0.2, random_state=42)


def train_model(model, model_name, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, artifact_path="model")

    print(f"{model_name} Accuracy: {acc:.4f}")
    return acc, model

def get_latest_run_id(experiment_name):
    from mlflow.entities import ViewType

    # Get the latest run from a specific experiment (replace with your experiment name or ID)
    experiment_name = "IrisModelExperiments"
    client = mlflow.tracking.MlflowClient()
    runs = client.search_runs(
        experiment_ids=client.get_experiment_by_name(experiment_name).experiment_id,
        order_by=["start_time DESC"],
        max_results=1,
        run_view_type=ViewType.ACTIVE_ONLY
    )

    if runs:
        latest_run = runs[0]
        print(f"Latest run ID: {latest_run.info.run_id}")
        # You can then use latest_run.info.run_id for downloading
        return latest_run.info.run_id
    else:
        print(f"No runs found for experiment: {experiment_name}")
        return None

def get_latest_modified_directory(path):
    import os
    # Get all entries in the path
    entries = [os.path.join(path, d) for d in os.listdir(path)]
    # Filter only directories
    directories = [d for d in entries if os.path.isdir(d)]
    if not directories:
        return None
    # Get the latest modified directory
    latest_dir = max(directories, key=os.path.getmtime)
    return os.path.basename(latest_dir)

def copy_latest_run_artifacts():
    import shutil
    import os

    # source_mlruns_path = "mlruns/1/models/" + get_latest_modified_directory("mlruns/1/models") + "/artifacts"  # Or "mlruns/1/SOME_RUN_ID/artifacts/models"
    source_mlruns_path = "mlruns/1/" + get_latest_modified_directory("mlruns/1") + "/artifacts/model"  # Or "mlruns/1/SOME_RUN_ID/artifacts/models"
    destination_local_path = "mlflow_model"

    # Create the destination directory if it doesn't exist
    os.makedirs(destination_local_path, exist_ok=True)

    try:
        # Use shutil.copytree for copying a directory
        # If 'models' is a directory containing your model files (e.g., MLmodel, model.pkl)
        shutil.copytree(source_mlruns_path, destination_local_path, dirs_exist_ok=True)
        print(f"Successfully copied '{source_mlruns_path}' to '{destination_local_path}'")

    except FileNotFoundError:
        print(f"Error: Source path '{source_mlruns_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("IrisModelExperiments")

    X_train, X_test, y_train, y_test = load_and_preprocess()

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
    }

    best_acc = 0
    best_model = None
    best_model_name = ""

    for name, model in models.items():
        acc, trained_model = train_model(model, name, X_train, X_test, y_train, y_test)
        if acc > best_acc:
            best_acc = acc
            best_model = trained_model
            best_model_name = name

    # Register best model
    with mlflow.start_run(run_name=best_model_name):
        mlflow.log_param("model_name",best_model_name)
        mlflow.log_metric("accuracy", best_acc)
        mlflow.sklearn.log_model(best_model,artifact_path="model", registered_model_name=best_model_name)

    copy_latest_run_artifacts()

if __name__ == "__main__":
    main()

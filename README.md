# mlops-iris-classification
 Building a minimal but complete MLOps pipeline for an ML model using a well-known open dataset.

 ## Directories
**.dvc:** This directory contains the internal information for DVC (Data Version Control). It's used to track large data files and machine learning models without checking them into Git.

**.github/workflows**:This folder holds your CI/CD pipeline configurations for GitHub Actions. These YAML files automate tasks like testing, building Docker images, and deploying your application.

**data**:This directory stores the datasets used for your project. It can contain raw data, processed data, and feature-engineered files that your model trains on.

**mlflow_model**: This folder contains a machine learning model saved in the standard MLflow format. This makes the model easy to load, serve, or register in a model registry.

**mlruns/1/models**:This is a default directory created by MLflow to log experiment runs. It stores metrics, parameters, artifacts, and saved model versions from your training scripts.

**src**:The "source" directory contains the core Python modules for your project. This typically includes scripts for data processing, model training, and evaluation.

**templates**:This folder holds the HTML templates for your web application. Frameworks like Flask or FastAPI use these files to render the user interface.

**tests**:This directory contains all the test scripts for your project. It's used for unit testing and integration testing to ensure your code is reliable.

## Files
**.dvcignore**:Similar to .gitignore, this file tells DVC which files or directories it should not track. This is useful for excluding temporary files or logs from data versioning.

**.gitignore**:This file specifies which files and directories Git should ignore. It's used to keep your repository clean by excluding things like environment files, caches, and large data files.

**Dockerfile**:This is a text file that contains instructions to build a Docker image for your application. It defines the environment, dependencies, and commands needed to containerize your project.

**README.md**:
This is the documentation file for your project, written in Markdown. It usually includes a project description, setup instructions, and usage examples.

**app.py**:This is typically the main entry point for your web application. It likely contains the Flask or FastAPI code to load your model and expose prediction endpoints.

**mlflow.db**:This is the backend database file, usually SQLite, where MLflow stores all the metadata for your experiments. It tracks parameters, metrics, and artifact locations for every run.

**requirements.txt**:This file lists all the Python packages and their specific versions required for the project. It allows anyone to easily replicate the development environment using pip install -r requirements.txt.

**scaler.pkl**:This is a serialized Python object, likely a Scikit-learn scaler (like MinMaxScaler or StandardScaler). It's saved after being fitted on the training data and is used to preprocess new data before making predictions.

# Use a slim Python base image for smaller image size
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy requirements.txt first to leverage Docker's build cache
COPY requirements.txt .
# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the rest of your Flask application files to the WORKDIR
# Since WORKDIR is /app, 'COPY . .' is equivalent to 'COPY . /app/'
# and is often less problematic.
COPY . .

# Copy your MLflow model directory into the container
# Assuming your MLflow model is in a directory named 'mlflow_model' relative to your Dockerfile
#COPY ./mlflow_model ./mlflow_model # Copies to /app/mlflow_model

# Expose the port your Flask app listens on (e.g., 5000)
EXPOSE 8000

# Command to run your Flask application
CMD ["python", "app.py"]
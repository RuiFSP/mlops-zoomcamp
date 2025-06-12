FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY model_utils.py taxi_prediction_flow.py ./

# Create directory for test data 
RUN mkdir -p data/raw models mlflow_data

# Export environment variable for MLflow
ENV MLFLOW_TRACKING_URI=http://mlflow:5000

# Command to run when the container starts
CMD ["python", "taxi_prediction_flow.py"]

# NYC Taxi Duration Prediction Workflow with Prefect

This project implements an end-to-end ML pipeline for predicting NYC taxi trip durations using Prefect for orchestration and MLflow for experiment tracking.

## Table of Contents

- [NYC Taxi Duration Prediction Workflow with Prefect](#nyc-taxi-duration-prediction-workflow-with-prefect)
  - [Table of Contents](#table-of-contents)
  - [Overview](#overview)
  - [Requirements](#requirements)
  - [Project Structure](#project-structure)
  - [Setup and Installation](#setup-and-installation)
  - [Setup and Running](#setup-and-running)
    - [MLflow Setup](#mlflow-setup)
      - [Option 1: Local MLflow Server](#option-1-local-mlflow-server)
      - [Option 2: MLflow with Docker](#option-2-mlflow-with-docker)
    - [Running the Flow](#running-the-flow)
    - [Deploying Workflows](#deploying-workflows)
      - [1. Start the Prefect Server](#1-start-the-prefect-server)
      - [2. Create a Work Pool](#2-create-a-work-pool)
      - [3. Create and Deploy the Workflow](#3-create-and-deploy-the-workflow)
      - [4. Starting a Worker](#4-starting-a-worker)
  - [Workflow Architecture](#workflow-architecture)
    - [Tasks](#tasks)
    - [Reliability Features](#reliability-features)
  - [Managing Deployments](#managing-deployments)
    - [View Deployment Details](#view-deployment-details)
    - [Manage Schedules](#manage-schedules)
  - [Docker Services Management](#docker-services-management)
  - [Integrated MLOps Workflow](#integrated-mlops-workflow)
  - [Advanced Usage](#advanced-usage)
    - [Custom Experiment Tracking](#custom-experiment-tracking)
      - [Creating Custom Experiments](#creating-custom-experiments)
      - [Nested Runs](#nested-runs)
    - [Model Registry Integration](#model-registry-integration)
      - [Registering a Model](#registering-a-model)
      - [Managing Model Lifecycle](#managing-model-lifecycle)
    - [CI/CD Integration](#cicd-integration)
      - [GitHub Actions Example](#github-actions-example)
    - [Monitoring and Observability](#monitoring-and-observability)
      - [Health Checks](#health-checks)
      - [Metrics Visualization](#metrics-visualization)
    - [Advanced Troubleshooting](#advanced-troubleshooting)
      - [Database Issues](#database-issues)
      - [Container Networking](#container-networking)
  - [Troubleshooting](#troubleshooting)
  - [MLOps Enhancements](#mlops-enhancements)
    - [End-to-End Testing](#end-to-end-testing)
    - [Monitoring and Observability](#monitoring-and-observability-1)
    - [Visualization](#visualization)
    - [Infrastructure Improvements](#infrastructure-improvements)
    - [Benefits of These Enhancements](#benefits-of-these-enhancements)
    - [Next Steps for MLOps Enhancement](#next-steps-for-mlops-enhancement)
  - [Homework](#homework)
    - [Question 1. Select the Tool](#question-1-select-the-tool)
    - [Question 2. Version](#question-2-version)
    - [Question 3. Creating a pipeline](#question-3-creating-a-pipeline)
    - [Question 4. Data preparation](#question-4-data-preparation)

## Overview

This project demonstrates the use of Prefect for orchestrating an ML pipeline that:
1. Loads NYC yellow taxi data
2. Processes and prepares features
3. Trains a linear regression model
4. Evaluates model performance 
5. Logs results to MLflow

The workflow is designed to run automatically on a monthly schedule, using training data from two months ago and validation data from the previous month.

## Requirements

- Python 3.9+
- Prefect 3.x (`pip install prefect>=3.0.0`)
- MLflow (`pip install mlflow`)
- scikit-learn, pandas, numpy, etc. (see project dependencies)
- NYC taxi dataset files in parquet format

## Project Structure

- `taxi_prediction_flow.py` - Main workflow implementation with Prefect
- `model_utils.py` - Utility functions for data preparation and model training
- `data/` - Directory containing yellow taxi data files
- `mlflow_monitor.py` - Script for monitoring MLflow server health
- `visualize_metrics.py` - Utility for visualizing experiment metrics
- `docker-services.sh` - Helper script for managing Docker containers
- `mlops-workflow.sh` - All-in-one workflow management script

## Setup and Installation

This project uses Python 3.9+ and the following dependencies:

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

## Setup and Running

### MLflow Setup

#### Option 1: Local MLflow Server

Start the MLflow tracking server locally:

```bash
# Navigate to the project directory
cd /mlops-zoomcamp-2025/03-orchestration

# Start the MLflow server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

#### Option 2: MLflow with Docker

You can run MLflow in Docker containers:

```bash
# Navigate to the project directory
cd /mlops-zoomcamp-2025/03-orchestration

# Create the directory for MLflow data
mkdir -p mlflow_data

# Start MLflow using the helper script
./docker-services.sh mlflow-only

# Check logs
./docker-services.sh logs mlflow
```

### Running the Flow

To run the workflow once locally:

```bash
python taxi_prediction_flow.py
```

### Deploying Workflows

#### 1. Start the Prefect Server

```bash
prefect server start
```

#### 2. Create a Work Pool

```bash
prefect work-pool create --type process "taxi-prediction-pool"
```

#### 3. Create and Deploy the Workflow

```bash
python taxi_prediction_flow.py --deploy
```

#### 4. Starting a Worker

```bash
PREFECT_API_URL=http://127.0.0.1:4200/api prefect worker start --pool taxi-prediction-pool
```

## Workflow Architecture

### Tasks

1. **Get Data Periods** - Determines the appropriate time periods for training and validation data
2. **Prepare Training Data** - Loads and preprocesses training data
3. **Prepare Validation Data** - Loads and preprocesses validation data
4. **Train Model** - Trains a linear regression model on the prepared data
5. **Evaluate Model** - Calculates RMSE on validation data
6. **Log Model** - Logs the model, metrics, and parameters to MLflow

### Reliability Features

- **Retries** - Critical tasks have retry mechanisms
- **Error Handling** - Comprehensive error handling with detailed logging
- **Task-level Logging** - Each task includes proper logging for debugging and monitoring
- **Parameter Validation** - Data validation to ensure inputs meet expectations

## Managing Deployments

### View Deployment Details

```bash
prefect deployment ls
prefect deployment inspect "NYC Taxi Duration Prediction/monthly-taxi-duration"
```

### Manage Schedules

```bash
# Pause a schedule
prefect deployment schedule pause "NYC Taxi Duration Prediction/monthly-taxi-duration"

# Resume a schedule
prefect deployment schedule resume "NYC Taxi Duration Prediction/monthly-taxi-duration"

# Run a deployment on-demand
prefect deployment run "NYC Taxi Duration Prediction/monthly-taxi-duration"
```

## Docker Services Management

The project includes helper scripts to manage Docker containers:

```bash
# View all available options
./docker-services.sh --help

# Start all services
./docker-services.sh start

# Start only MLflow
./docker-services.sh mlflow-only

# Start only Prefect workflow
./docker-services.sh prefect-only

# Check logs
./docker-services.sh logs

# Stop all services
./docker-services.sh stop

# Rebuild the Docker images
./docker-services.sh build

# Clean up containers, volumes and data
./docker-services.sh clean
```

## Integrated MLOps Workflow

For a more streamlined experience, use the all-in-one MLOps workflow script:

```bash
# Show available commands
./mlops-workflow.sh help

# Setup the environment
./mlops-workflow.sh setup

# Start all services
./mlops-workflow.sh start

# Check status of all services
./mlops-workflow.sh status

# Run end-to-end workflow test
./mlops-workflow.sh test

# Open a shell in a container
./mlops-workflow.sh shell mlflow

# Start monitoring MLflow server
./mlops-workflow.sh monitor
```

## Advanced Usage

### Custom Experiment Tracking

#### Creating Custom Experiments

To create and use custom experiments in MLflow instead of the default "nyc-taxi-experiment":

```python
import mlflow

# Set the tracking URI
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

# Create a custom experiment
experiment_name = "my-custom-experiment" 
try:
    experiment_id = mlflow.create_experiment(
        experiment_name,
        artifact_location="./artifacts",
        tags={"version": "v1", "team": "ml-team"}
    )
except:
    # If experiment already exists, get its ID
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id

# Set as active experiment
mlflow.set_experiment(experiment_name)

# Use in your workflow
with mlflow.start_run():
    # Your training code here
    mlflow.log_param("param1", value1)
    mlflow.log_metric("metric1", score)
```

#### Nested Runs

For more complex workflows, you can use nested runs to organize your experiments:

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

# Parent run for the entire pipeline
with mlflow.start_run(run_name="full_pipeline") as parent_run:
    parent_run_id = parent_run.info.run_id
    mlflow.log_param("pipeline_version", "v1.0")
    
    # Child run for data preparation
    with mlflow.start_run(run_name="data_preparation", nested=True) as prep_run:
        mlflow.log_param("data_source", "s3://my-bucket/taxi-data")
        mlflow.log_metric("data_quality_score", 0.95)
    
    # Child run for model training
    with mlflow.start_run(run_name="model_training", nested=True) as train_run:
        mlflow.log_param("model_type", "linear_regression")
        mlflow.log_metric("rmse", 5.43)

    # Child run for model evaluation
    with mlflow.start_run(run_name="model_evaluation", nested=True) as eval_run:
        mlflow.log_metric("test_rmse", 5.89)
        mlflow.log_metric("r2", 0.82)
```

### Model Registry Integration

The MLflow Model Registry provides a centralized model store and APIs to programmatically manage the full model lifecycle.

#### Registering a Model

To register a model in the MLflow Model Registry:

```python
import mlflow
from mlflow.models import ModelSignature
from mlflow.types import Schema, ColSpec

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("nyc-taxi-experiment")

# Define model input and output schema for validation
input_schema = Schema([
    ColSpec("float", "PU_DO"),
    ColSpec("float", "trip_distance"),
    ColSpec("integer", "passenger_count")
])
output_schema = Schema([ColSpec("float")])
signature = ModelSignature(inputs=input_schema, outputs=output_schema)

# Log and register the model
with mlflow.start_run():
    # ... your model training code ...
    
    # Log the model with signature and register it
    mlflow.sklearn.log_model(
        model,
        "model",
        signature=signature,
        registered_model_name="nyc-taxi-duration-predictor"
    )
```

#### Managing Model Lifecycle

Transition a model version through stages (Staging, Production, Archived):

```python
from mlflow.tracking import MlflowClient

client = MlflowClient("http://localhost:5000")

# Get the latest model version
model_name = "nyc-taxi-duration-predictor"
latest_version = max([int(mv.version) for mv in client.search_model_versions(f"name='{model_name}'")])

# Transition to staging with description
client.transition_model_version_stage(
    name=model_name,
    version=latest_version,
    stage="Staging",
    archive_existing_versions=True  # Archive any existing models in this stage
)

# Add description and tags
client.update_model_version(
    name=model_name,
    version=latest_version,
    description="Model trained on January 2023 data"
)

client.set_model_version_tag(
    name=model_name,
    version=latest_version,
    key="data_source",
    value="yellow_taxi_2023_01"
)
```

### CI/CD Integration

This section covers integrating MLflow and Prefect with CI/CD pipelines.

#### GitHub Actions Example

Here's an example GitHub Actions workflow file (`.github/workflows/train-and-deploy.yml`):

```yaml
name: Train and Deploy Model

on:
  schedule:
    - cron: '0 0 1 * *'  # Run at midnight on the 1st of each month
  workflow_dispatch:  # Allow manual triggers

jobs:
  train:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Start MLflow server
      run: |
        mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db &
        sleep 5  # Give server time to start
    
    - name: Run training workflow
      run: python taxi_prediction_flow.py
      env:
        MLFLOW_TRACKING_URI: http://localhost:5000
    
    - name: Upload MLflow database as artifact
      uses: actions/upload-artifact@v3
      with:
        name: mlflow-database
        path: mlflow.db
```

### Monitoring and Observability

#### Health Checks

The MLflow server includes a health check endpoint for monitoring:

```bash
# Check MLflow server health
curl http://localhost:5000/health
```

The health endpoint returns:
- HTTP 200 if the server is healthy
- HTTP 503 if the server is unhealthy

#### Metrics Visualization

Use the `visualize_metrics.py` script for custom visualizations of your MLflow metrics:

```bash
# Generate metric visualization
python visualize_metrics.py --experiment-name "nyc-taxi-experiment" --metric "rmse"
```

### Advanced Troubleshooting

#### Database Issues

If you encounter SQLite database locking issues:

```bash
# Backup current database
cp mlflow.db mlflow.db.backup

# Check for database corruption
sqlite3 mlflow.db "PRAGMA integrity_check;"

# Vacuum the database to optimize
sqlite3 mlflow.db "VACUUM;"
```

#### Container Networking

If containers can't communicate:

```bash
# Check container network
docker network inspect bridge

# Get container IP addresses
docker inspect -f '{{.Name}} - {{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' $(docker ps -q)

# Test connectivity between containers
docker exec mlflow ping -c 2 prefect
```

## Troubleshooting

For scheduled runs to work correctly:

1. Keep the Prefect server running
2. Keep a worker running
3. Ensure data availability

Common Issues:

- If scheduled runs aren't working, check that the Prefect server and worker are running
- If MLflow isn't accessible, check Docker container health and network connectivity

## MLOps Enhancements

### End-to-End Testing

- Created `test_workflow.py` script to perform complete workflow testing
- Implemented validation of MLflow experiment logging from Prefect
- Added data file verification and Docker environment setup

### Monitoring and Observability

- Enhanced MLflow server with a dedicated health check endpoint
- Created `mlflow_monitor.py` for monitoring server health and metrics
- Added Flask server in MLflow container that provides `/health` endpoint
- Updated Docker healthcheck configuration to use the new health endpoint

### Visualization

- Created `visualize_metrics.py` script for generating insights from MLflow metrics
- Implemented metric history visualization
- Added parameter impact analysis
- Created tools for comparing multiple metrics across runs

### Infrastructure Improvements

- Enhanced MLflow Dockerfile with health endpoints
- Updated Docker Compose configuration for better reliability
- Added proper health check mechanisms
- Updated dependencies in requirements.txt to support all new features

### Benefits of These Enhancements

1. **Improved Reliability**: Better health checks and monitoring ensure services stay up and running
2. **Enhanced Observability**: New tools provide insights into system performance and model metrics
3. **Simplified Workflow**: All-in-one management scripts make the system easier to use
4. **Better Documentation**: Comprehensive guides for both basic and advanced usage
5. **Data-Driven Insights**: Visualization tools help extract meaningful patterns from experiments

### Next Steps for MLOps Enhancement

1. **Automated Alerting**: Add email/Slack notifications for failed runs or degraded service health
2. **Model Performance Drift Detection**: Implement monitoring for model performance degradation
3. **A/B Testing Framework**: Create infrastructure for comparing models in production
4. **Resource Usage Optimization**: Add monitoring for container resource usage and optimization
5. **Multi-Model Ensemble**: Implement capabilities for ensemble modeling using the best runs

 

This is what we used (adjusted for yellow dataset):

```python
def read_dataframe(filename):
    df = pd.read_parquet(filename)

    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df.duration = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].astype(str)
```

## Homework

The goal of this homework is to create a simple training pipeline, use mlflow to track experiments and register best model, but use Mage for it.

We'll use [the same NYC taxi dataset](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page), the **Yellow** taxi data for March, 2023. 

### Question 1. Select the Tool

You can use the same tool you used when completing the module,
or choose a different one for your homework.

What's the name of the orchestrator you chose? 


### Question 2. Version

What's the version of the orchestrator? 


### Question 3. Creating a pipeline

Let's read the March 2023 Yellow taxi trips data.

How many records did we load? 

- 3,003,766
- 3,203,766
- 3,403,766
- 3,603,766

(Include a print statement in your code)

### Question 4. Data preparation

Let's continue with pipeline creation.

We will use the same logic for preparing the data we used previously.

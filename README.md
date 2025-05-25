# MLOps Zoomcamp - My Assignments

This repository contains my solutions and assignments for the MLOps Zoomcamp course by DataTalks.Club. Each module of the course is organized into separate directories with their respective assignments, code, and documentation.

## Course Overview

MLOps Zoomcamp is a comprehensive course on Machine Learning Operations (MLOps) covering the full lifecycle of ML projects, from data preparation to model deployment and monitoring.

## Repository Structure

- `01-intro/` 
  - Introduction to MLOps
  - NYC Yellow taxi trip duration prediction model
  - Data preprocessing, feature engineering, and model training
  - Metrics evaluation on training and validation datasets

- `02-experiment-tracking/`
  - Experiment tracking with MLflow
  - Training and hyperparameter tuning for taxi trip duration prediction
  - MLflow model registry and tracking server setup
  - Green Taxi Trip data processing and model optimization

## Completed Assignments

### Module 1: Introduction to MLOps

In this module, I created a linear regression model to predict the duration of taxi trips in NYC using Yellow Taxi Trip Records from January and February 2023. The assignment involved:

- Loading and exploring Yellow Taxi Trip data
- Computing trip durations and handling outliers
- Implementing one-hot encoding for categorical features
- Training a linear regression model
- Evaluating the model on both training and validation datasets

Tools & Libraries used:
- Python
- Pandas, NumPy
- scikit-learn
- Jupyter notebooks

### Module 2: Experiment Tracking and Model Registry

In this module, I implemented experiment tracking and model management using MLflow. The assignment focused on:

- Setting up MLflow for experiment tracking and model versioning
- Processing NYC Green Taxi Trip data for January to March 2023
- Training RandomForestRegressor models with MLflow autologging
- Running hyperparameter optimization with hyperopt
- Setting up a local MLflow tracking server with SQLite backend
- Managing model lifecycle using MLflow Model Registry
- Evaluating and selecting the best model based on validation metrics

Tools & Libraries used:
- MLflow (version 1.27.0)
- Hyperopt for hyperparameter tuning
- scikit-learn (RandomForestRegressor)
- Pandas, NumPy
- SQLite for MLflow backend storage

## Setup and Installation

This project uses Python 3.10+ and the following dependencies as defined in the `pyproject.toml` file:


The project uses the `uv` package manager for faster dependency resolution and installation:

```bash
# Create a virtual environment using uv
uv venv
source .venv/bin/activate  # On Linux/Mac

# Install dependencies with uv
uv pip install -e .

# The dependencies are locked in the uv.lock file for reproducibility
```

## Future Modules

I'll be adding more modules as I progress through the course:
- Module 3: Orchestration and ML pipelines
- Module 4: Model deployment
- Module 5: Model monitoring
- Module 6: Best practices

## Contact

Feel free to reach out if you have any questions about my assignments or solutions.
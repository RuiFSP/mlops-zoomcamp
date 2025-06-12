# Configuration Files

This directory contains configuration files for the NYC Taxi Duration Prediction workflow project:

- `docker-compose.yml` - Docker Compose configuration for setting up the MLflow and Prefect services
- `mlflow.dockerfile` - Dockerfile for building the MLflow tracking server with health monitoring
- `prefect.dockerfile` - Dockerfile for setting up the Prefect workflow environment
- `deployment.yaml` - Prefect deployment configuration template

## Docker Compose

The `docker-compose.yml` file defines two main services:
- MLflow tracking server for experiment tracking
- Prefect server and worker for workflow orchestration

## Dockerfiles

### MLflow Dockerfile

The `mlflow.dockerfile` includes:
- Base Python environment setup
- MLflow installation and dependencies
- Health check endpoint configuration
- Volume mounts for persistent storage
- Network settings for service discovery

### Prefect Dockerfile

The `prefect.dockerfile` includes:
- Python environment with Prefect installed
- Project dependencies from requirements.txt
- Configuration for running as a Prefect worker
- Volume mounts for data and model access

## Deployment Configuration

The `deployment.yaml` template is used by Prefect for:
- Deployment name and metadata
- Scheduling information
- Infrastructure requirements
- Parameter defaults

For more information, see the main project README.md file.

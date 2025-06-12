# Utility Scripts

This directory contains utility scripts for the NYC Taxi Duration Prediction workflow project:

- `docker-services.sh` - Helper script for managing Docker containers
- `mlops-workflow.sh` - All-in-one workflow management script
- `mlflow-docker.sh` - Script specifically for managing MLflow Docker container
- `mlflow_monitor.py` - Python script for monitoring MLflow server health
- `visualize_metrics.py` - Utility for visualizing experiment metrics from MLflow

## Usage

These scripts are typically called from the project root using the symbolic links provided:

```bash
# From project root
./docker-services.sh start
./mlops-workflow.sh status
```

Or they can be called directly from this directory:

```bash
# From scripts directory
./docker-services.sh start
./mlflow_monitor.py --continuous
```

For more information, see the main project README.md file.

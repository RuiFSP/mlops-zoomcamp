# 3. Orchestration and ML Pipelines

## Notes

### Running the Homework Notebook with MLflow

To run the `homework03.ipynb` notebook with MLflow tracking:

#### Step 1: Start MLflow Server

First, start the MLflow tracking server in a terminal:

```bash
# Navigate to the project directory
cd /home/ruifspinto/projects/mlops-zoomcamp-2025/03-orchestration

# Start the MLflow server
mlflow server --host 0.0.0.0 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

This will:
- Start an MLflow server accessible at http://localhost:5000
- Store tracking data in a SQLite database file named `mlflow.db`
- Automatically create an `mlartifacts` directory for storing model artifacts

> **Important**: Keep this terminal window open while running the notebook.

#### Step 2: Run the Jupyter Notebook

In a new terminal window:

```bash
# Navigate to the project directory
cd /mlops-zoomcamp-2025/03-orchestration

# Start Jupyter Notebook
jupyter notebook
```

Then open the `homework03.ipynb` notebook in your browser.

The notebook is configured to automatically connect to the MLflow server at http://localhost:5000. If the connection fails, it will fall back to using local file storage in the `mlruns` directory.

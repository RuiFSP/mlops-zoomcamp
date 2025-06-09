FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install mlflow==2.12.1 Flask==2.2.3 waitress==2.1.2

# Create health check script
RUN mkdir -p /app
COPY <<'EOF' /app/health_endpoint.py
#!/usr/bin/env python
# Simple health check endpoint for MLflow server

import os
import sys
import threading
import time
import requests
from flask import Flask, jsonify
from waitress import serve

app = Flask(__name__)
MLFLOW_URL = "http://localhost:5000"
health_status = {"status": "initializing", "mlflow_accessible": False}

def update_health_status():
    """Background thread to check MLflow status"""
    global health_status
    while True:
        try:
            # Check MLflow API
            response = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/experiments/list")
            health_status = {
                "status": "healthy" if response.status_code == 200 else "degraded",
                "mlflow_accessible": response.status_code == 200,
                "experiments_count": len(response.json().get("experiments", [])) if response.status_code == 200 else 0
            }
        except Exception as e:
            health_status = {
                "status": "unhealthy", 
                "mlflow_accessible": False,
                "error": str(e)
            }
        time.sleep(30)  # Check every 30 seconds

@app.route('/health')
def health():
    """Health endpoint that returns current status"""
    return jsonify(health_status)

if __name__ == '__main__':
    # Start background thread for health checks
    thread = threading.Thread(target=update_health_status, daemon=True)
    thread.start()
    
    # Start web server on port 8080
    print("Starting health endpoint on port 8080")
    serve(app, host="0.0.0.0", port=8080)
EOF

RUN chmod +x /app/health_endpoint.py

EXPOSE 5000 8080

# Use a script to start both MLflow and the health endpoint
COPY <<'EOF' /app/start.sh
#!/bin/bash
set -e

# Start health endpoint in background
python /app/health_endpoint.py &

# Start MLflow server in foreground
mlflow server \
    --backend-store-uri sqlite:///home/mlflow_data/mlflow.db \
    --host 0.0.0.0 \
    --port 5000
EOF

RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]

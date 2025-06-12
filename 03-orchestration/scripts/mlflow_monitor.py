#!/usr/bin/env python
# coding: utf-8

import os
import sys
import json
import time
import logging
import requests
import argparse
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MLflowMonitor:
    """
    A simple monitoring tool for MLflow server.
    Provides health checks and basic metrics about MLflow server.
    """

    def __init__(self, mlflow_uri="http://localhost:5000", output_file=None):
        self.mlflow_uri = mlflow_uri
        self.output_file = output_file
        self.health_status = False
        self.experiments_count = 0
        self.runs_count = 0
        self.latest_activity = None

    def check_health(self):
        """Check if MLflow server is up and running"""
        try:
            start_time = time.time()
            response = requests.get(f"{self.mlflow_uri}/health", timeout=5)
            response_time = time.time() - start_time

            if response.status_code == 200:
                self.health_status = True
                logger.info(
                    f"MLflow server is healthy. Response time: {response_time:.4f}s"
                )
                return True, response_time
            else:
                self.health_status = False
                logger.error(
                    f"MLflow server returned status code {response.status_code}"
                )
                return False, response_time

        except requests.RequestException as e:
            self.health_status = False
            logger.error(f"Failed to connect to MLflow server: {e}")
            return False, None

    def collect_metrics(self):
        """Collect basic metrics about MLflow server"""
        if not self.health_status:
            logger.error("Cannot collect metrics, MLflow server is not healthy")
            return False

        try:
            # Get experiments
            response = requests.get(
                f"{self.mlflow_uri}/api/2.0/experiments/list", timeout=5
            )
            if response.status_code == 200:
                experiments = response.json().get("experiments", [])
                self.experiments_count = len(experiments)
                logger.info(f"Found {self.experiments_count} experiments")

                # Count runs across all experiments
                total_runs = 0
                latest_time = None

                for exp in experiments:
                    exp_id = exp["experiment_id"]
                    runs_response = requests.get(
                        f"{self.mlflow_uri}/api/2.0/experiments/get?experiment_id={exp_id}",
                        timeout=5,
                    )

                    if runs_response.status_code == 200:
                        # Process runs if available
                        if "runs" in runs_response.json():
                            exp_runs = runs_response.json().get("runs", [])
                            total_runs += len(exp_runs)

                            # Check for latest activity
                            for run in exp_runs:
                                end_time = run.get("info", {}).get("end_time")
                                if end_time and (
                                    latest_time is None or end_time > latest_time
                                ):
                                    latest_time = end_time

                self.runs_count = total_runs

                if latest_time:
                    # Convert timestamp to datetime
                    self.latest_activity = datetime.fromtimestamp(latest_time / 1000.0)
                    latest_str = self.latest_activity.strftime("%Y-%m-%d %H:%M:%S")
                else:
                    latest_str = "No activity found"

                logger.info(f"Total runs: {total_runs}")
                logger.info(f"Latest activity: {latest_str}")
                return True
            else:
                logger.error(f"Failed to get experiments: {response.status_code}")
                return False

        except Exception as e:
            logger.error(f"Error collecting metrics: {e}")
            return False

    def save_status(self):
        """Save health status and metrics to a file"""
        if not self.output_file:
            logger.info("No output file specified, skipping save")
            return

        data = {
            "timestamp": datetime.now().isoformat(),
            "health_status": self.health_status,
            "experiments_count": self.experiments_count,
            "runs_count": self.runs_count,
            "latest_activity": self.latest_activity.isoformat()
            if self.latest_activity
            else None,
            "mlflow_uri": self.mlflow_uri,
        }

        try:
            with open(self.output_file, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"Status saved to {self.output_file}")
        except Exception as e:
            logger.error(f"Failed to save status: {e}")

    def run_check(self, save=True):
        """Run a complete health check"""
        health, response_time = self.check_health()

        if health:
            metrics = self.collect_metrics()

            if save:
                self.save_status()

            return health and metrics

        return False


def main():
    parser = argparse.ArgumentParser(description="MLflow Server Health Monitor")
    parser.add_argument(
        "--uri",
        default=os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        help="MLflow tracking server URI (default: http://localhost:5000)",
    )
    parser.add_argument("--output", help="Path to save the health status JSON file")
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Run in continuous mode, checking every 5 minutes",
    )
    args = parser.parse_args()

    monitor = MLflowMonitor(args.uri, args.output)

    if args.continuous:
        logger.info(f"Starting continuous monitoring of MLflow server at {args.uri}")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                monitor.run_check()
                time.sleep(300)  # Check every 5 minutes
        except KeyboardInterrupt:
            logger.info("Monitoring stopped by user")
    else:
        success = monitor.run_check()
        return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

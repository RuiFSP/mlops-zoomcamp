#!/usr/bin/env python
# coding: utf-8

"""
MLflow Metrics Visualizer

This script fetches metrics from MLflow and creates visualizations to help
analyze experiment results.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime


def setup_mlflow_connection(tracking_uri=None):
    """Set up connection to MLflow tracking server"""
    if tracking_uri is None:
        tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")

    print(f"Connecting to MLflow at: {tracking_uri}")
    mlflow.set_tracking_uri(tracking_uri)

    try:
        # Test connection
        client = MlflowClient()
        experiments = client.search_experiments()
        print(
            f"Successfully connected to MLflow. Found {len(experiments)} experiments."
        )
        return client
    except Exception as e:
        print(f"Failed to connect to MLflow: {e}")
        sys.exit(1)


def get_experiment_by_name(client, experiment_name):
    """Get experiment by name, creating it if it doesn't exist"""
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        print(f"Experiment '{experiment_name}' not found.")
        return None

    print(f"Found experiment '{experiment_name}' (ID: {experiment.experiment_id})")
    return experiment


def fetch_runs_data(client, experiment_id):
    """Fetch all runs for an experiment and convert to DataFrame"""
    runs = client.search_runs(experiment_ids=[experiment_id])

    if not runs:
        print("No runs found for this experiment.")
        return None

    print(f"Found {len(runs)} runs.")

    # Extract run data into a list of dictionaries
    runs_data = []
    for run in runs:
        run_data = {
            "run_id": run.info.run_id,
            "start_time": datetime.fromtimestamp(run.info.start_time / 1000.0),
            "status": run.info.status,
            **run.data.params,
            **run.data.metrics,
        }
        runs_data.append(run_data)

    # Convert to DataFrame
    df = pd.DataFrame(runs_data)
    return df


def plot_metric_history(df, metric_name, save_path=None):
    """Plot the history of a metric over time"""
    if metric_name not in df.columns:
        print(f"Metric '{metric_name}' not found in the data.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(df["start_time"], df[metric_name], "o-", linewidth=2, markersize=8)
    plt.title(f"{metric_name.upper()} over Time", fontsize=15)
    plt.xlabel("Run Date", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)

    # Add average line
    avg = df[metric_name].mean()
    plt.axhline(y=avg, color="r", linestyle="--", label=f"Average: {avg:.4f}")

    # Add annotations for min and max
    min_idx = df[metric_name].idxmin()
    max_idx = df[metric_name].idxmax()

    plt.annotate(
        f"Min: {df[metric_name].min():.4f}",
        xy=(df["start_time"].iloc[min_idx], df[metric_name].min()),
        xytext=(10, -30),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )

    plt.annotate(
        f"Max: {df[metric_name].max():.4f}",
        xy=(df["start_time"].iloc[max_idx], df[metric_name].max()),
        xytext=(10, 30),
        textcoords="offset points",
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"),
    )

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def plot_param_impact(df, param_name, metric_name, save_path=None):
    """Plot the impact of a parameter on a metric"""
    if param_name not in df.columns or metric_name not in df.columns:
        print(f"Parameter '{param_name}' or metric '{metric_name}' not found.")
        return

    plt.figure(figsize=(10, 6))

    # Check if parameter is numeric
    if pd.api.types.is_numeric_dtype(df[param_name]):
        # For numeric parameters, use a scatter plot
        plt.scatter(df[param_name], df[metric_name], alpha=0.7, s=100)

        # Add trend line
        z = np.polyfit(df[param_name].astype(float), df[metric_name], 1)
        p = np.poly1d(z)
        plt.plot(
            df[param_name].astype(float),
            p(df[param_name].astype(float)),
            "r--",
            alpha=0.7,
            linewidth=2,
        )

    else:
        # For categorical parameters, use a box plot
        sns.boxplot(x=param_name, y=metric_name, data=df)
        plt.xticks(rotation=45)

    plt.title(f"Impact of {param_name} on {metric_name}", fontsize=15)
    plt.xlabel(param_name, fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def compare_metrics(df, metrics, save_path=None):
    """Compare multiple metrics across runs"""
    available_metrics = [m for m in metrics if m in df.columns]

    if not available_metrics:
        print("None of the specified metrics are available.")
        return

    # Normalize metrics to compare on same scale
    normalized_df = pd.DataFrame()
    for metric in available_metrics:
        normalized_df[metric] = (df[metric] - df[metric].min()) / (
            df[metric].max() - df[metric].min()
        )

    normalized_df["Run"] = range(1, len(df) + 1)

    # Melt the DataFrame for seaborn plotting
    melted_df = pd.melt(
        normalized_df,
        id_vars=["Run"],
        value_vars=available_metrics,
        var_name="Metric",
        value_name="Normalized Value",
    )

    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=melted_df, x="Run", y="Normalized Value", hue="Metric", marker="o"
    )

    plt.title("Comparison of Normalized Metrics Across Runs", fontsize=15)
    plt.xlabel("Run Number", fontsize=12)
    plt.ylabel("Normalized Value (0-1)", fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(title="Metric")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def print_best_runs(df, metric_name, n=5, minimize=True):
    """Print the N best runs for a given metric"""
    if metric_name not in df.columns:
        print(f"Metric '{metric_name}' not found in the data.")
        return

    # Sort by metric (ascending if minimize, descending otherwise)
    sorted_df = df.sort_values(by=metric_name, ascending=minimize)

    print(f"\n{'Best' if minimize else 'Top'} {n} runs by {metric_name}:")
    print("-" * 80)

    # Print only relevant columns
    columns_to_print = ["run_id", "start_time", metric_name]

    # Add param columns if they exist
    param_cols = [
        col
        for col in df.columns
        if col not in ["run_id", "start_time", "status"]
        and col not in df.select_dtypes(include=[np.number]).columns
    ]
    columns_to_print.extend(param_cols)

    # Print top N runs
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    print(sorted_df[columns_to_print].head(n).to_string(index=False))
    print("-" * 80)


def main():
    parser = argparse.ArgumentParser(description="MLflow Metrics Visualizer")
    parser.add_argument("--uri", default=None, help="MLflow tracking URI")
    parser.add_argument(
        "--experiment",
        default="nyc-taxi-experiment",
        help="Experiment name (default: nyc-taxi-experiment)",
    )
    parser.add_argument(
        "--metric", default="rmse", help="Metric to visualize (default: rmse)"
    )
    parser.add_argument(
        "--param", default=None, help="Parameter to analyze against the metric"
    )
    parser.add_argument("--compare", nargs="+", help="List of metrics to compare")
    parser.add_argument(
        "--output-dir", default="./plots", help="Directory to save plots"
    )

    args = parser.parse_args()

    # Setup MLflow connection
    client = setup_mlflow_connection(args.uri)

    # Get experiment
    experiment = get_experiment_by_name(client, args.experiment)
    if experiment is None:
        return

    # Fetch runs data
    df = fetch_runs_data(client, experiment.experiment_id)
    if df is None:
        return

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # Generate visualizations
    metric = args.metric

    # Plot metric history
    plot_metric_history(df, metric, save_path=f"{args.output_dir}/{metric}_history.png")

    # If parameter is specified, plot parameter impact
    if args.param:
        plot_param_impact(
            df,
            args.param,
            metric,
            save_path=f"{args.output_dir}/{args.param}_impact_on_{metric}.png",
        )

    # If compare metrics are specified, plot comparison
    if args.compare:
        compare_metrics(
            df, args.compare, save_path=f"{args.output_dir}/metrics_comparison.png"
        )

    # Print best runs by the specified metric
    print_best_runs(df, metric, n=5, minimize=True)

    print("\nVisualization complete!")


if __name__ == "__main__":
    main()

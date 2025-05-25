import os
import pickle
import click
import mlflow
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from math import sqrt

def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
@click.option(
    "--experiment_name",
    default="nyc-taxi-experiment-homework",
    help="Name of the MLflow experiment"
)
def run_train(data_path: str, experiment_name: str):
    # Set tracking URI
    mlflow.set_tracking_uri("file:./mlruns")
    
    # Set the experiment name
    mlflow.set_experiment(experiment_name)
    
    # Load the data
    X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
    X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))
    
    # Start MLflow run
    with mlflow.start_run():
        # Train a RandomForestRegressor with default parameters
        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        # Calculate RMSE
        rmse = sqrt(mean_squared_error(y_val, y_pred))
        print(f"RMSE: {rmse}")
        
        # Log parameters manually instead of using autologging
        mlflow.log_param("max_depth", rf.max_depth)
        mlflow.log_param("n_estimators", rf.n_estimators)
        mlflow.log_param("min_samples_split", rf.min_samples_split)
        mlflow.log_metric("rmse", rmse)
        
        # Skip model logging to avoid setuptools/distutils issues
        # mlflow.sklearn.log_model(rf, "model")
        
        # Print parameters to easily see min_samples_split
        print(f"\nParameters:")
        print(f"  min_samples_split: {rf.min_samples_split}")
        print(f"  max_depth: {rf.max_depth}")
        print(f"  n_estimators: {rf.n_estimators}")

if __name__ == '__main__':
    run_train()
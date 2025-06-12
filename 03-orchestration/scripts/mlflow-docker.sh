#!/bin/bash

# Script to manage the MLflow Docker environment

function show_help {
    echo "MLflow Docker Environment Manager"
    echo ""
    echo "Usage:"
    echo "  $0 [command]"
    echo ""
    echo "Commands:"
    echo "  start      - Start the MLflow Docker container"
    echo "  stop       - Stop the MLflow Docker container"
    echo "  restart    - Restart the MLflow Docker container"
    echo "  logs       - Show logs from the MLflow container"
    echo "  status     - Check if MLflow container is running"
    echo "  help       - Show this help message"
    echo ""
}

function start_mlflow {
    echo "Starting MLflow container..."
    mkdir -p mlflow_data
    docker compose up -d mlflow
    echo "MLflow should be accessible at http://localhost:5000"
}

function stop_mlflow {
    echo "Stopping MLflow container..."
    docker compose down mlflow
}

function restart_mlflow {
    echo "Restarting MLflow container..."
    docker compose restart mlflow
    echo "MLflow should be accessible at http://localhost:5000"
}

function show_logs {
    echo "Showing MLflow container logs (press Ctrl+C to exit)..."
    docker compose logs -f mlflow
}

function check_status {
    if docker compose ps | grep -q mlflow-server; then
        echo "MLflow container is running"
        echo "MLflow UI is available at: http://localhost:5000"
    else
        echo "MLflow container is not running"
    fi
}

# Main logic
case "$1" in
    start)
        start_mlflow
        ;;
    stop)
        stop_mlflow
        ;;
    restart)
        restart_mlflow
        ;;
    logs)
        show_logs
        ;;
    status)
        check_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        ;;
esac

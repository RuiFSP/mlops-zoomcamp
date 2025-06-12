#!/bin/bash
# Docker helper script for MLOps project

# Function to display usage information
show_usage() {
    echo "Usage: $0 [OPTIONS] COMMAND"
    echo ""
    echo "A helper script to manage Docker services for the MLOps workflow"
    echo ""
    echo "Options:"
    echo "  -h, --help        Show this help message"
    echo ""
    echo "Commands:"
    echo "  start             Start all services (MLflow and Prefect)"
    echo "  stop              Stop all services"
    echo "  restart           Restart all services"
    echo "  mlflow-only       Start only MLflow service"
    echo "  prefect-only      Start only Prefect service"
    echo "  logs [service]    Show logs for all services or a specific service"
    echo "  build             Rebuild all Docker images"
    echo "  clean             Remove containers, volumes and data"
    echo ""
    echo "Examples:"
    echo "  $0 start         # Start all services"
    echo "  $0 logs mlflow   # Show MLflow logs"
}

# Check if Docker and Docker Compose are installed
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! docker compose version &> /dev/null; then
        echo "Docker Compose is not available. Please install Docker Compose."
        exit 1
    fi
}

# Start services
start_services() {
    echo "Starting Docker services..."
    docker compose up -d "$@"
    echo "Services started. MLflow UI available at http://localhost:5000"
}

# Stop services
stop_services() {
    echo "Stopping Docker services..."
    docker compose down
    echo "Services stopped."
}

# Show logs
show_logs() {
    if [ -z "$1" ]; then
        docker compose logs --follow
    else
        docker compose logs --follow "$1"
    fi
}

# Clean everything
clean_all() {
    echo "Warning: This will remove all containers, volumes, and data."
    read -p "Are you sure you want to continue? (y/N): " confirm
    
    if [[ $confirm == [yY] || $confirm == [yY][eE][sS] ]]; then
        docker compose down -v
        echo "Removing data directories..."
        rm -rf mlflow_data/*
        echo "Clean completed."
    else
        echo "Clean operation cancelled."
    fi
}

# Main script logic
check_docker

case "$1" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        stop_services
        start_services
        ;;
    mlflow-only)
        start_services mlflow
        ;;
    prefect-only)
        start_services prefect
        ;;
    logs)
        show_logs "$2"
        ;;
    build)
        docker compose build
        echo "Docker images rebuilt."
        ;;
    clean)
        clean_all
        ;;
    -h|--help)
        show_usage
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

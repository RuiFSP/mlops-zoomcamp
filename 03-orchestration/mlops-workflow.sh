#!/bin/bash
# Full MLOps workflow deployment script
# This script provides a comprehensive interface for managing
# the MLflow and Prefect deployment for the NYC Taxi Duration Prediction project

set -e  # Exit on error

# Colors for better UI
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Banner
show_banner() {
    clear
    echo -e "${BLUE}==============================================${NC}"
    echo -e "${BLUE}  NYC Taxi Duration Prediction MLOps Workflow ${NC}"
    echo -e "${BLUE}==============================================${NC}"
    echo ""
}

# Help menu
show_help() {
    echo -e "${GREEN}Usage:${NC} $0 [command]"
    echo ""
    echo "Commands:"
    echo -e "  ${YELLOW}setup${NC}           Initial setup (create directories, download data)"
    echo -e "  ${YELLOW}start${NC}           Start all services"
    echo -e "  ${YELLOW}stop${NC}            Stop all services"
    echo -e "  ${YELLOW}status${NC}          Check status of all services"
    echo -e "  ${YELLOW}logs${NC} [service]  View logs (mlflow, prefect, or all)"
    echo -e "  ${YELLOW}rebuild${NC}         Rebuild Docker images"
    echo -e "  ${YELLOW}clean${NC}           Remove all containers, volumes, and data"
    echo -e "  ${YELLOW}test${NC}            Run end-to-end workflow test"
    echo -e "  ${YELLOW}monitor${NC}         Start monitoring MLflow server"
    echo -e "  ${YELLOW}shell${NC} [service] Open a shell in a container"
    echo ""
    echo "Examples:"
    echo "  $0 setup      # Set up initial directories and download data"
    echo "  $0 start      # Start all services"
    echo "  $0 test       # Run end-to-end test"
    echo ""
}

# Check if Docker is available
check_docker() {
    if ! command -v docker &> /dev/null; then
        echo -e "${RED}Error: Docker is not installed or not in PATH${NC}"
        exit 1
    fi
    
    if ! docker compose version &> /dev/null; then
        echo -e "${RED}Error: Docker Compose is not available${NC}"
        exit 1
    fi
}

# Setup function - create directories and download data if needed
setup() {
    echo -e "${BLUE}Setting up project environment...${NC}"
    
    # Create required directories
    mkdir -p data/raw models mlflow_data
    
    # Check if we need to download data files
    if [ ! -f "data/yellow_tripdata_2023-01.parquet" ] || [ ! -f "data/yellow_tripdata_2023-02.parquet" ]; then
        echo -e "${YELLOW}Downloading NYC Taxi data files...${NC}"
        
        # Download files
        for month in "01" "02"; do
            if [ ! -f "data/yellow_tripdata_2023-${month}.parquet" ]; then
                echo "Downloading data for 2023-${month}..."
                curl -sSL "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-${month}.parquet" \
                    -o "data/yellow_tripdata_2023-${month}.parquet"
            fi
        done
    else
        echo -e "${GREEN}Data files already exist.${NC}"
    fi
    
    # Check and make scripts executable
    for script in test_workflow.py mlflow_monitor.py docker-services.sh; do
        if [ -f "$script" ]; then
            chmod +x "$script"
        fi
    done
    
    echo -e "${GREEN}Setup complete!${NC}"
}

# Start services
start() {
    echo -e "${BLUE}Starting services...${NC}"
    ./docker-services.sh start
    
    # Wait for MLflow to be available
    echo -e "${YELLOW}Waiting for MLflow server to be ready...${NC}"
    attempt=1
    max_attempts=10
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s http://localhost:5000 > /dev/null; then
            echo -e "${GREEN}MLflow server is ready!${NC}"
            echo -e "${GREEN}Services started successfully!${NC}"
            echo -e "MLflow UI: ${BLUE}http://localhost:5000${NC}"
            echo -e "MLflow Health: ${BLUE}http://localhost:8080/health${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}Waiting for MLflow server (attempt $attempt/$max_attempts)...${NC}"
        sleep 5
        ((attempt++))
    done
    
    echo -e "${RED}MLflow server did not start properly. Check logs with '$0 logs mlflow'${NC}"
    return 1
}

# Check status of services
status() {
    echo -e "${BLUE}Checking service status...${NC}"
    
    # Print a nice status table
    echo -e "${YELLOW}CONTAINER          STATUS          PORTS${NC}"
    echo -e "${YELLOW}------------------------------------------------${NC}"
    
    # MLflow
    mlflow_status=$(docker ps --filter "name=mlflow-server" --format "{{.Status}}" 2>/dev/null || echo "Not running")
    if [[ $mlflow_status == *"Up"* ]]; then
        echo -e "MLflow Server     ${GREEN}Running${NC}        5000, 8080"
        # Try to check health status
        health_status=$(curl -s http://localhost:8080/health 2>/dev/null || echo '{"status":"unknown"}')
        mlflow_health=$(echo $health_status | grep -o '"status":"[^"]*' | cut -d'"' -f4)
        echo -e "  - Health Status: ${BLUE}$mlflow_health${NC}"
        
        # Check experiments count
        exp_count=$(curl -s http://localhost:5000/api/2.0/mlflow/experiments/list | grep -o '"experiments":\[[^]]*\]' | grep -o 'experiment_id' | wc -l)
        runs_count=$(curl -s http://localhost:5000/api/2.0/mlflow/runs/search | grep -o '"runs":\[[^]]*\]' | grep -o 'run_id' | wc -l)
        echo -e "  - Experiments: ${BLUE}$exp_count${NC}, Runs: ${BLUE}$runs_count${NC}"
    else
        echo -e "MLflow Server     ${RED}Not running${NC}"
    fi
    
    # Prefect
    prefect_status=$(docker ps --filter "name=prefect-workflow" --format "{{.Status}}" 2>/dev/null || echo "Not running")
    if [[ $prefect_status == *"Up"* ]]; then
        echo -e "Prefect Workflow  ${GREEN}Running${NC}"
    else
        echo -e "Prefect Workflow  ${RED}Not running${NC}"
    fi
    
    echo ""
}

# Run tests
run_tests() {
    echo -e "${BLUE}Running end-to-end workflow test...${NC}"
    python ./test_workflow.py
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Tests completed successfully!${NC}"
    else
        echo -e "${RED}Tests failed. Check the output for details.${NC}"
        return 1
    fi
}

# Start monitoring
monitor() {
    echo -e "${BLUE}Starting MLflow monitoring...${NC}"
    
    # Run the monitoring script in continuous mode
    python ./mlflow_monitor.py --continuous
}

# Open shell in container
open_shell() {
    local service=$1
    
    if [ -z "$service" ]; then
        echo -e "${RED}Error: You must specify a service (mlflow or prefect)${NC}"
        return 1
    fi
    
    if [ "$service" = "mlflow" ]; then
        container="mlflow-server"
    elif [ "$service" = "prefect" ]; then
        container="prefect-workflow"
    else
        echo -e "${RED}Error: Unknown service '$service'. Use 'mlflow' or 'prefect'.${NC}"
        return 1
    fi
    
    echo -e "${BLUE}Opening shell in $container container...${NC}"
    docker exec -it $container bash || docker exec -it $container sh
}

# Main execution
check_docker

# Show banner
show_banner

# Parse command
case "$1" in
    setup)
        setup
        ;;
    start)
        start
        ;;
    stop)
        echo -e "${BLUE}Stopping services...${NC}"
        ./docker-services.sh stop
        echo -e "${GREEN}Services stopped.${NC}"
        ;;
    status)
        status
        ;;
    logs)
        echo -e "${BLUE}Showing logs...${NC}"
        ./docker-services.sh logs $2
        ;;
    rebuild)
        echo -e "${BLUE}Rebuilding Docker images...${NC}"
        ./docker-services.sh build
        echo -e "${GREEN}Rebuild complete.${NC}"
        ;;
    clean)
        echo -e "${BLUE}Cleaning up resources...${NC}"
        ./docker-services.sh clean
        echo -e "${GREEN}Clean complete.${NC}"
        ;;
    test)
        run_tests
        ;;
    monitor)
        monitor
        ;;
    shell)
        open_shell $2
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        show_help
        ;;
esac

exit 0

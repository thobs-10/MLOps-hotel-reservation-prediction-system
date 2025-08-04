#!/bin/bash

# Function to build the package
build_package() {
  echo "Building the package and installing dependencies..."
  uv build
  uv pip install -e .
}

setup_zenml() {
  echo "Setting up ZenML..."
  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
  zenml login --local
}

setup_pandera() {
  echo "Setting up Pandera for data validation..."
  export DISABLE_PANDERA_IMPORT_WARNING=True

}

# Function to start Feast UI
start_feast() {
  echo "Starting Feast UI..."
  cd feature_store/
  feast ui
  cd .. # Return to the root directory
}

# Function to start MLflow server
start_mlflow() {
  echo "Starting MLflow server..."
  mlflow server --host 127.0.0.1 --port 8085
}

# Function to run the pipelines
run_pipelines() {
  echo "Running data ingestion, feature engineering, and model training pipelines..."
  build_package
  setup_zenml
  setup_pandera
  # start_feast
  start_mlflow
  sleep 5 # Wait for Feast and MLflow to start
  python src/run_pipelines.py
}
# function to spin up the docker compose
setup_docker_compose() {
  echo "Setting up Docker Compose..."
  docker compose up --build
}
# Function to run pre-commit checks
run_pre_commit() {
  echo "Running pre-commit checks..."
  SKIP=no-commit-to-branch pre-commit run --all-files
}
# clean codebase
clean() {
  echo "Cleaning the codebase..."
  find . -type d -name ".ruff_cache" -exec rm -rf {} +
  find . -type d -name ".pytest_cache" -exec rm -rf {} +
  find . -type d -name ".mypy_cache" -exec rm -rf {} +
  find . -type d -name ".ipynb_checkpoints" -exec rm -rf {} +
  find . -type d -name "cachedir" -exec rm -rf {} +
  find . -type d -name "*.egg-info" -exec rm -rf {} +
  echo "Codebase cleaned."
}

# For CI: Run specific functions if arguments are provided
if [ $# -gt 0 ]; then
    for func in "$@"; do
        $func
    done
    exit 0
fi

# Main execution
echo "Starting the development process..."

# Run the pipelines
run_pipelines
sleep 5
# Run pre-commit checks
run_pre_commit
# Set up Docker Compose
setup_docker_compose
sleep 10
start_feast
# Clean the codebase
clean
echo "Development process completed."

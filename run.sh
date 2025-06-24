#!/bin/bash

# Function to install the package
install_package() {
  echo "Installing the package..."
  pip install .
}

setup_zenml() {
  echo "Setting up ZenML..."
  export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
  zenml login --local
}

# Function to start Feast UI
start_feast() {
  echo "Starting Feast UI..."
  cd my_feature_store/feature_repo
  feast ui
  cd ../.. # Return to the root directory
}

# Function to start MLflow server
start_mlflow() {
  echo "Starting MLflow server..."
  mlflow server --host 127.0.0.1 --port 8085
}

# Function to run the pipelines
run_pipelines() {
  echo "Running data ingestion, feature engineering, and model training pipelines..."
  python run_pipelines.py
}

# Function to run pre-commit checks
run_pre_commit() {
  echo "Running pre-commit checks..."
  SKIP=no-commit-to-branch pre-commit --files src/
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

# Install the package
install_package

# Start feast ui and mlflow in the background
setup_zenml &
start_feast &
start_mlflow &

# Wait for feast and mlflow to finish starting up.
sleep 5

# Run the pipelines
run_pipelines

# Run pre-commit checks
run_pre_commit
clean
echo "Development process completed."

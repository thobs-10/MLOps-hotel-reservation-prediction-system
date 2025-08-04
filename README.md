
# MLOps Hotel Reservation Cancellation Prediction System

## Project Overview

Hotel companies often struggle to anticipate when customers will cancel their reservations, resulting in lost revenue and missed opportunities for proactive engagement. This project aims to build an end-to-end machine learning system that predicts reservation cancellations, enabling hotels to reach out to customers before they cancel and optimize their operations.

The solution leverages historical booking data and customer behavior to forecast cancellation risk. By integrating AI-driven predictions into hotel workflows, the system helps maximize occupancy, improve customer retention, and increase revenue.


## Problem Statement

Hotels lose money and operational efficiency when reservations are cancelled unexpectedly. The goal is to develop a predictive model and supporting infrastructure that:
- Identifies bookings at risk of cancellation
- Provides actionable insights for customer engagement
- Enables automated monitoring and retraining
- Follows MLOps best practices for reproducibility, scalability, and maintainability
## Lifecycle of the project
### 1. Understanding the Problem Statement
Define the project scope and requirements to accurately predict hotel reservation cancellation.

### 2. Data Collection
Download hotel data that contains information about bookings and customer preferences.

### 3. Exploratory Data Analysis (EDA)
Analyze the dataset to uncover patterns and identify vital factors contributing to reservation cancellation.

### 4. Data Ingestion and Feature Engineering
Handle missing values, engineer relevant features, and prepare data for modeling by storing in a feature store location.

### 5. Model Training and Tracking
Train different models for accurate predictions through rigorus experiments and tracking those experiments.

### 6. Model Hyperparameter Tuning
Optimize the chosen model to enhance performance.

### 7. Model Evaluation
Evaluate the model using metrics such as accuracy, F1-score, and precision.

### 8. Model Registry
Save the final model to model registry in for deployment.

### 9. Model Deployment
Containerize the model and make it accessible via an API endpoint.

### 10. Model Serving
Expose the model through a REST API for real-time feedback prediction.

### 11. Continuous Integration/Continuous Deployment (CI/CD)
Automate pipeline workflows to ensure reproducibility and scalability.

### 12 Best Practices
Apply best practices to make sure the codebase is clean, maintainable, testable and easy to run.


## Technologies Used

- **Python 3.11**: Core language for all scripts and pipelines
- **UV**: For project management and virtual environments
- **Jupyter Notebooks**: For exploratory data analysis and prototyping
- **ZenML**: For data and ML pipeline orchestration
- **Feast**: Feature store for managing and serving features
- **MLflow**: Experiment tracking and model registry
- **FastAPI**: Web service for real-time predictions
- **Docker & Docker Compose**: Containerization and orchestration
- **Prometheus & Grafana**: Monitoring and dashboarding
- **pandas, scikit-learn, xgboost**: Data processing and modeling
- **loguru**: Logging
- **pre-commit, Bandit**: Code quality (formatting, linting) and security
- **GitHub Actions**: CI/CD pipeline
- **pytest**: Testing framework
- **Bash**: For scripting and automation tasks
- **Dockerhub**: For container image hosting

## Tool Explanation
- **ZenML**: Orchestrates the entire ML pipeline, ensuring reproducibility and scalability. For more details, see [ZenML Documentation](https://docs.zenml.io/).
- **Feast**: Manages and serves features for model training and inference. More information can be found in the [Feast Documentation](https://docs.feast.dev/).
- **Prometheus & Grafana**: Used for monitoring model performance and system metrics. Prometheus scrapes metrics from the FastAPI application, while Grafana provides visualization dashboards. For more details, see [Prometheus Documentation](https://prometheus.io/docs/introduction/overview/) and [Grafana Documentation](https://grafana.com/docs/grafana/latest/).


## Project Criteria and Achievements

| Criteria                               | Status |
| -------------------------------------- | :----: |
| Problem description                    |   ✅    |
| Cloud                                  |   ❎    |
| Experiment tracking and model registry |   ✅    |
| Workflow orchestration                 |   ✅    |
| Model Deployment                       |   ✅    |
| Reproducibility                        |   ✅    |
| Model Monitoring                       |   ✅    |
| Best Practices                         |   ✅    |


## How to Run the Project

### 1. Prepare the Dataset
- Download the hotel reservation dataset from [hotel reservation dataset](https://www.kaggle.com/datasets/ahsan81/hotel-reservations-classification-dataset).
- Place the file in `data/raw/`.

### 2. Environment Setup
- Ensure you are using Python 3.11 (if not ,recommend using: uv).
- Create and activate a virtual environment:
  ```
  uv venv
  source venv/bin/activate
  ```

### 3. Configure Environment Variables
- Create a `.env` file in the project root with:
  ```
    RAW_DATA_PATH="data/raw/Hotel Reservations.csv"
    PROCESSED_DATA_PATH="data/processed_data/"
    FEATURES_PATH="data/feature_store/"
    MODEL_ARTIFACTS_PATH="src/models/artifacts/"
    HOST=<your_host>
    PORT=<your_port>
    PREDICTION_LOG_PATH="src/logs/prediction_logs.csv"
    API_URL="<local_host>:<your_port>/predict"
  ```

### 4. Run The System
- From the `root` folder, run:
  ```
   ./run.sh
  ```
- Then run this command on a new terminal to start feast UI instance for feature store:
  ```
  ./run.sh start_feast
  ```
- **Optional** - Individual pipelines can be run from `src/pipeline/`.
- Logs will show progress.
- Links(will show once the system is done running):
  - [Prometheus](http://localhost:9090/metrics)
  - [Grafana](http://localhost:3000) (default credentials: admin/admin)
  - [Feast](http://localhost:8888)
  - [MLflow](http://localhost:8085)
  - [ZenML](http://localhost:8237)


### 6. Test the API Endpoint and Make Predictions
- Use the provided `input_request_data_simulation.py` script and check the logs for predictions and grafana dashboard for model performance.
- To simulate input requests, run:
  ```
  python input_request_data_simulation.py
  ```

### 7. Monitoring and Alerting
- Prometheus scrapes metrics from FastAPI (`/metrics` endpoint or `/targets` to see if the instance is up and running).
- Grafana dashboards visualize model performance and drift.
- Alerts can be configured for metric thresholds and retraining triggers.

### 8. CI/CD Integration
- GitHub Actions automate testing, linting, building, and deployment.
- Pre-commit hooks enforce code quality and security.
- Docker images are deployed to Dockerhub and can be pulled:
  ```
  docker pull thobela10/hotel-reservation-project:latest
  docker run -p 8000:8000 thobela10/hotel-reservation-project
  ```

## Best Practices Implemented

- Unit tests
- Linter and code formatter
- Bash for automation tasks
- Pre-commit hooks
- CI/CD pipeline



This README provides all necessary information to understand, run, and evaluate the hotel reservation cancellation prediction system.

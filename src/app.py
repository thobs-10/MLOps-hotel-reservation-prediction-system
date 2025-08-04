import os
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI, Response
from loguru import logger
from prometheus_client import Counter, Histogram, generate_latest
import pandas as pd
from src.pipeline.inference_pipeline import (
    BookingRequest,
    BookingResponse,
    InferencePipeline,
)
from src.utils.model_retriain_utils import monitor_drift_and_accuracy


load_dotenv()

app = FastAPI()

# Prometheus metrics
REQUEST_COUNT = Counter("prediction_requests_total", "Total prediction requests")
PREDICTION_LATENCY = Histogram(
    "prediction_latency_seconds", "Prediction latency in seconds"
)

# Store requests and predictions for monitoring/retraining
PREDICTION_LOG_PATH = os.getenv("PREDICTION_LOG_PATH", "src/logs/prediction_logs.csv")


def log_prediction(request: BookingRequest, prediction: BookingResponse):
    """Log request and prediction to a CSV file for monitoring/retraining."""
    try:
        # Convert request and prediction to dicts
        req_dict = (
            request.model_dump() if hasattr(request, "model_dump") else dict(request)
        )
        pred_dict = (
            prediction.model_dump()
            if hasattr(prediction, "model_dump")
            else dict(prediction)
        )
        log_entry = {**req_dict, **pred_dict}
        # Append to CSV
        df = pd.DataFrame([log_entry])
        if not os.path.exists(PREDICTION_LOG_PATH):
            df.to_csv(PREDICTION_LOG_PATH, index=False)
        else:
            df.to_csv(PREDICTION_LOG_PATH, mode="a", header=False, index=False)
    except Exception as e:
        logger.error(f"Failed to log prediction: {e}")


@app.get("/")
async def read_root() -> Dict[str, str]:
    return {"message": "Welcome to the Hotel Booking Prediction API"}


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")


@app.post("/predict", response_model=BookingResponse)
async def predict_booking(request: BookingRequest) -> BookingResponse:
    REQUEST_COUNT.inc()
    with PREDICTION_LATENCY.time():
        try:
            inference_pipeline = InferencePipeline()
            input_df = await inference_pipeline.preprocess_data(request)
            prediction = await inference_pipeline.predict(input_df)
            logger.info(f"Prediction made: {prediction}")
            if not prediction:
                raise ValueError("Prediction returned empty result.")
            log_prediction(request, prediction)
            return prediction
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise e


def check_model_performance_and_trigger_retraining():
    """
    Example function: Check logged predictions and trigger retraining if needed.
    You can implement drift/accuracy checks here and call your retraining pipeline.
    """
    if not os.path.exists(PREDICTION_LOG_PATH):
        logger.info("No prediction logs found.")
        return
    df = pd.read_csv(PREDICTION_LOG_PATH)

    drift_threshold = 0.1
    accuracy_threshold = 0.8

    # Log metrics
    data_drift_score, accuracy = monitor_drift_and_accuracy(
        reference_df=df,
        current_df=df,
        reference_preds=df["reference_preds"],
        current_preds=df["current_preds"],
        y_true=df["y_true"],
        y_pred=df["y_pred"],
    )

    # Example retraining trigger
    if data_drift_score > drift_threshold or accuracy < accuracy_threshold:
        logger.warning("Drift or accuracy threshold breached. Triggering retraining...")
        # retrain_model()
    else:
        logger.info("Model performance within acceptable range.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", 8000)),
        reload=True,
    )

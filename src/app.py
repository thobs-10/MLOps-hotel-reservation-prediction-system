import os
from typing import Dict

from dotenv import load_dotenv
from fastapi import FastAPI
from loguru import logger

from src.pipeline.inference_pipeline import (
    BookingRequest,
    BookingResponse,
    InferencePipeline,
)

# Load environment variables
load_dotenv()


app = FastAPI()


@app.get("/")
async def read_root() -> Dict[str, str]:
    return {"message": "Welcome to the Hotel Booking Prediction API"}


@app.post("/predict", response_model=BookingResponse)
async def predict_booking(request: BookingRequest) -> BookingResponse:
    """
    Predict booking status based on the provided booking details.

    Args:
        request (BookingRequest): Booking details for prediction.

    Returns:
        BookingResponse: Predicted booking status.
    """
    try:
        inference_pipeline = InferencePipeline()
        input_df = await inference_pipeline.preprocess_data(request)
        prediction = await inference_pipeline.predict(input_df)
        logger.info(f"Prediction made: {prediction}")
        if not prediction:
            raise ValueError("Prediction returned empty result.")
        return prediction
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ["HOST"],
        port=int(os.environ["PORT"]),
        reload=True,
    )

import os

import joblib
import pandas as pd
from loguru import logger
from pydantic import BaseModel
from sklearn.pipeline import Pipeline

from src.entity.config_entity import ModelTrainingConfig


class BookingRequest(BaseModel):
    lead_time: float
    avg_price_per_room: float
    no_of_special_requests: int
    arrival_date: int
    arrival_month: int


class BookingResponse(BaseModel):
    booking_status: int


class InferencePipeline:
    def __init__(self) -> None:
        self.model_path = os.path.join(
            ModelTrainingConfig.model_registry_path, "random_forest.pkl"
        )

    def load_model(self) -> Pipeline:
        """Load the trained model from the specified path.

        Raises:
            ValueError: If the model path is not set.
            HTTPException: If the model file is not found.

        Returns:
            Pipeline: The loaded model pipeline.
        """
        try:
            if not self.model_path:
                raise ValueError("Model path is not set.")
            self.model = joblib.load(self.model_path)
            return self.model
        except FileNotFoundError as e:
            logger.error(f"Model file not found: {e}")
            raise e

    async def preprocess_data(self, data: BookingRequest) -> pd.DataFrame:
        """Preprocess the input data for prediction.
        Args:
            data (BookingRequest): Input booking request data.
        Returns:
            pd.DataFrame: Preprocessed DataFrame ready for prediction.
        """
        input_data = {
            "lead_time": data.lead_time,
            "avg_price_per_room": data.avg_price_per_room,
            "no_of_special_requests": data.no_of_special_requests,
            "arrival_date": data.arrival_date,
            "arrival_month": data.arrival_month,
        }
        input_df = pd.DataFrame([input_data])
        return input_df

    async def predict(self, data: pd.DataFrame) -> BookingResponse:
        """Make a prediction using the loaded model.
        Args:
            data (pd.DataFrame): Preprocessed input data.

        Returns:
            BookingResponse: Prediction result containing booking status.
        """
        model = self.load_model()
        try:
            predictions = model.predict(data)
            return BookingResponse(booking_status=int(predictions[0]))
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise e

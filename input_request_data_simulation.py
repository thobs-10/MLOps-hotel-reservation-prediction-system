import os
import random
import time
from typing import Dict

import requests
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

API_URL = os.environ.get("API_URL", "http://0.0.0.0:8000/predict")


def generate_random_request() -> Dict[str, int | float]:
    """
    Generate a random input payload for hotel booking prediction.
    """
    return {
        "lead_time": random.randint(1, 60),  # nosec
        "avg_price_per_room": round(random.uniform(50, 500), 2),  # nosec
        "no_of_special_requests": random.randint(0, 5),  # nosec
        "arrival_date": random.randint(1, 31),  # nosec
        "arrival_month": random.randint(1, 12),  # nosec
    }


def simulate_batch_requests(batch_size=50, delay=0.2) -> None:
    """
    Simulate a batch of requests to the FastAPI endpoint.
    """
    logger.info(f"🚀 Sending {batch_size} requests to {API_URL}")
    success, fail = 0, 0
    for i in range(batch_size):
        payload = generate_random_request()
        try:
            response = requests.post(API_URL, json=payload)
            if response.status_code == 200:
                logger.info(f"[{i + 1}/{batch_size}] ✅ Success: {response.json()}")
                success += 1
            else:
                logger.error(
                    f"[{i + 1}/{batch_size}] ❌ Status: {response.status_code}"
                )
                fail += 1
        except Exception as e:
            logger.error(f"[{i + 1}/{batch_size}] ❌ Error: {e}")
            fail += 1
        time.sleep(delay)
    logger.info(f"Batch complete: {success} success, {fail} failed.")


def main():
    simulate_batch_requests(batch_size=50, delay=0.2)


if __name__ == "__main__":
    main()

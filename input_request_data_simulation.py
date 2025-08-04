import requests
import time
import random
from loguru import logger

API_URL = "http://0.0.0.0:8000/predict"


def generate_random_request():
    """
    Generate a random input payload for hotel booking prediction.
    """
    return {
        "lead_time": random.randint(1, 60),
        "avg_price_per_room": round(random.uniform(50, 500), 2),
        "no_of_special_requests": random.randint(0, 5),
        "arrival_date": random.randint(1, 31),
        "arrival_month": random.randint(1, 12),
    }


def simulate_batch_requests(batch_size=50, delay=0.2):
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

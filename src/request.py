import requests
from loguru import logger


# quick test to check the fastapi server is running
def test_fastapi_server():
    """Test FastAPI server is running."""
    logger.info("🏪 Testing FastAPI server...")
    # request to the root endpoint
    input_json = {
        "lead_time": 10.0,
        "avg_price_per_room": 100.0,
        "no_of_special_requests": 3,
        "arrival_date": 25,
        "arrival_month": 4,
    }
    try:
        response = requests.post("http://0.0.0.0:8000/predict", json=input_json)
        if response.status_code == 200:
            logger.info("✅ FastAPI server is running.")
        else:
            logger.error(
                f"❌ FastAPI server is not running. Status code: {response.status_code}"
            )
    except Exception as e:
        logger.error(f"Error occurred while testing FastAPI server: {e}")


def main():
    """Main function to run the test."""
    test_fastapi_server()


if __name__ == "__main__":
    main()

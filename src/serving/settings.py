from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()

class Settings(BaseModel):
    APP_NAME: str = os.getenv("APP_NAME", "chatgpt-trading")
    APP_ENV: str = os.getenv("APP_ENV", "dev")
    APP_PORT: int = int(os.getenv("APP_PORT", 8000))

    EXCHANGE_ID: str = os.getenv("EXCHANGE_ID", "binance")
    EXCHANGE_API_KEY: str = os.getenv("EXCHANGE_API_KEY", "")
    EXCHANGE_API_SECRET: str = os.getenv("EXCHANGE_API_SECRET", "")
    EXCHANGE_PASSWORD: str = os.getenv("EXCHANGE_PASSWORD", "")
    EXCHANGE_TESTNET: bool = os.getenv("EXCHANGE_TESTNET", "true").lower() == "true"

    POSTGRES_HOST: str = os.getenv("POSTGRES_HOST", "postgres")
    POSTGRES_PORT: int = int(os.getenv("POSTGRES_PORT", 5432))
    POSTGRES_USER: str = os.getenv("POSTGRES_USER", "trader")
    POSTGRES_PASSWORD: str = os.getenv("POSTGRES_PASSWORD", "traderpwd")
    POSTGRES_DB: str = os.getenv("POSTGRES_DB", "trading")

    REDIS_HOST: str = os.getenv("REDIS_HOST", "redis")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))

    MLFLOW_TRACKING_URI: str = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")

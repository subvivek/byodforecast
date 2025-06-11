from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "BYOD Forecast API"
    
    # AWS Settings
    AWS_ROLE_ARN: Optional[str] = os.getenv("AWS_ROLE_ARN")
    AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
    S3_BUCKET: str = os.getenv("S3_BUCKET", "byod-forecast-data")
    
    # Model Settings
    DEFAULT_FORECAST_HORIZON: int = 30
    TRAIN_TEST_SPLIT: float = 0.8
    RANDOM_SEED: int = 42
    
    # Hyperparameter Tuning
    N_TRIALS: int = 50
    TIMEOUT: int = 600  # 10 minutes
    
    class Config:
        case_sensitive = True

settings = Settings() 
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime

class ForecastRequest(BaseModel):
    s3_path: str = Field(..., description="Path to the CSV file in S3")
    forecast_type: Literal["arima", "lightgbm", "linear_regression"] = Field(
        default="arima",
        description="Type of forecasting model to use"
    )
    error_metric: Literal["mse", "mae", "rmse"] = Field(
        default="rmse",
        description="Error metric to use for model evaluation"
    )
    sku: Optional[str] = Field(None, description="Specific SKU to forecast")
    steps: Optional[int] = Field(None, description="Number of steps to forecast ahead")

class ForecastPoint(BaseModel):
    timestamp: datetime
    value: float
    lower_bound: float
    upper_bound: float

class ModelMetrics(BaseModel):
    model_name: str
    error_metric: str
    error_value: float
    training_time: float
    prediction_time: float

class ForecastResponse(BaseModel):
    forecast_points: List[ForecastPoint]
    metrics: ModelMetrics
    best_model: str
    quantiles: dict = Field(
        ...,
        description="Dictionary of quantile forecasts (P30, P50, P80, P90, P95)"
    )
    created_at: datetime = Field(default_factory=datetime.utcnow) 
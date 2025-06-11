from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.forecast import ForecastRequest, ForecastResponse
from app.services.forecast_service import ForecastService
from app.core.config import settings
import logging

router = APIRouter()
forecast_service = ForecastService()

@router.post("/forecast", response_model=ForecastResponse)
async def create_forecast(
    request: ForecastRequest,
    background_tasks: BackgroundTasks
):
    """
    Create a new forecast using specified models and parameters.
    """
    try:
        # Validate input
        if not request.s3_path:
            raise HTTPException(status_code=400, detail="S3 path is required")
        
        # Generate forecast
        forecast_result = await forecast_service.generate_forecast(
            s3_path=request.s3_path,
            forecast_type=request.forecast_type,
            error_metric=request.error_metric,
            sku=request.sku,
            steps=request.steps or settings.DEFAULT_FORECAST_HORIZON
        )
        
        return forecast_result
        
    except Exception as e:
        logging.error(f"Error generating forecast: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/models")
async def list_models():
    """
    List available forecasting models and their capabilities.
    """
    return {
        "models": [
            {
                "name": "arima",
                "description": "AutoRegressive Integrated Moving Average",
                "capabilities": ["point_forecast", "quantiles"],
                "hyperparameters": ["p", "d", "q"]
            },
            {
                "name": "lightgbm",
                "description": "Light Gradient Boosting Machine",
                "capabilities": ["point_forecast", "quantiles"],
                "hyperparameters": ["learning_rate", "n_estimators", "max_depth"]
            },
            {
                "name": "linear_regression",
                "description": "Linear Regression with feature engineering",
                "capabilities": ["point_forecast", "quantiles"],
                "hyperparameters": ["fit_intercept", "normalize"]
            }
        ]
    } 
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List
import boto3
import pandas as pd
from datetime import datetime
import os

app = FastAPI(title="BYODForecast API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

class ForecastRequest(BaseModel):
    s3_path: str
    forecast_type: str  # "fast" or "precise"
    error_metric: str  # "MAE", "RMSE", "MAPE", "SMAPE"
    sku: Optional[str] = None

class ForecastResponse(BaseModel):
    forecast_id: str
    status: str
    message: str
    results: Optional[dict] = None

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

@app.get("/{path:path}")
async def serve_frontend(path: str):
    if os.path.exists(f"app/static/{path}"):
        return FileResponse(f"app/static/{path}")
    return FileResponse("app/static/index.html")

@app.post("/api/forecast", response_model=ForecastResponse)
async def create_forecast(request: ForecastRequest):
    try:
        # Generate a unique forecast ID
        forecast_id = f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # TODO: Implement the actual forecasting logic
        # This will be implemented in subsequent steps
        
        return ForecastResponse(
            forecast_id=forecast_id,
            status="pending",
            message="Forecast request received and queued for processing"
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"message": f"Error processing forecast request: {str(e)}"}
        )

@app.get("/api/forecast/{forecast_id}")
async def get_forecast(forecast_id: str):
    # TODO: Implement forecast retrieval logic
    return {"message": "Forecast retrieval endpoint"}

@app.get("/api/metrics")
async def get_available_metrics():
    return {
        "metrics": [
            {"id": "MAE", "name": "Mean Absolute Error"},
            {"id": "RMSE", "name": "Root Mean Squared Error"},
            {"id": "MAPE", "name": "Mean Absolute Percentage Error"},
            {"id": "SMAPE", "name": "Symmetric Mean Absolute Percentage Error"}
        ]
    } 
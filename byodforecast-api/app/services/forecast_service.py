import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from prophet import Prophet
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import boto3
import json
from datetime import datetime

class ForecastService:
    def __init__(self):
        self.s3_client = boto3.client('s3')
        
    def _load_data_from_s3(self, s3_path: str) -> pd.DataFrame:
        """Load data from S3 bucket."""
        bucket = s3_path.split('/')[0]
        key = '/'.join(s3_path.split('/')[1:])
        
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(response['Body'])
        return df
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, metric: str) -> float:
        """Calculate specified error metric."""
        if metric == 'MAE':
            return mean_absolute_error(y_true, y_pred)
        elif metric == 'RMSE':
            return np.sqrt(mean_squared_error(y_true, y_pred))
        elif metric == 'MAPE':
            return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        elif metric == 'SMAPE':
            return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred))) * 100
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    
    def _train_arima(self, data: pd.DataFrame) -> Dict:
        """Train ARIMA model."""
        model = sm.tsa.ARIMA(data['value'], order=(1, 1, 1))
        results = model.fit()
        return {'model': results}
    
    def _train_prophet(self, data: pd.DataFrame) -> Dict:
        """Train Prophet model."""
        model = Prophet()
        df = data.rename(columns={'date': 'ds', 'value': 'y'})
        model.fit(df)
        return {'model': model}
    
    def _train_random_forest(self, data: pd.DataFrame) -> Dict:
        """Train Random Forest model."""
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        X = data[['feature1', 'feature2']]  # Adjust features as needed
        y = data['value']
        model.fit(X, y)
        return {'model': model}
    
    def _train_linear_regression(self, data: pd.DataFrame) -> Dict:
        """Train Linear Regression model."""
        model = LinearRegression()
        X = data[['feature1', 'feature2']]  # Adjust features as needed
        y = data['value']
        model.fit(X, y)
        return {'model': model}
    
    def generate_forecast(self, s3_path: str, forecast_type: str, error_metric: str, sku: Optional[str] = None) -> Dict:
        """Generate forecast using specified models."""
        try:
            # Load data
            data = self._load_data_from_s3(s3_path)
            
            if sku:
                data = data[data['sku'] == sku]
            
            # Train models
            models = {
                'arima': self._train_arima(data),
                'prophet': self._train_prophet(data),
                'random_forest': self._train_random_forest(data),
                'linear_regression': self._train_linear_regression(data)
            }
            
            # Generate predictions
            predictions = {}
            for model_name, model_dict in models.items():
                # TODO: Implement prediction logic for each model
                predictions[model_name] = {
                    'point_forecast': [],
                    'p30': [],
                    'p50': [],
                    'p80': [],
                    'p90': [],
                    'p95': []
                }
            
            # Calculate metrics and select best model
            best_model = None
            best_metric = float('inf')
            
            for model_name, preds in predictions.items():
                metric_value = self._calculate_metrics(
                    data['value'].values,
                    preds['point_forecast'],
                    error_metric
                )
                if metric_value < best_metric:
                    best_metric = metric_value
                    best_model = model_name
            
            return {
                'forecast_id': f"forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'best_model': best_model,
                'predictions': predictions[best_model],
                'metrics': {
                    error_metric: best_metric
                }
            }
            
        except Exception as e:
            raise Exception(f"Error generating forecast: {str(e)}") 
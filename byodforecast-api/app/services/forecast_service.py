import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import boto3
import json
from datetime import datetime, timedelta
import logging
import optuna
import lightgbm as lgb
from app.core.config import settings
import time

class ForecastService:
    def __init__(self):
        # Create a session using the IAM role
        session = boto3.Session()
        sts_client = session.client('sts')
        
        # Assume the role
        if settings.AWS_ROLE_ARN:
            assumed_role = sts_client.assume_role(
                RoleArn=settings.AWS_ROLE_ARN,
                RoleSessionName='BYODForecastSession'
            )
            credentials = assumed_role['Credentials']
            
            # Create S3 client with assumed role credentials
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=credentials['AccessKeyId'],
                aws_secret_access_key=credentials['SecretAccessKey'],
                aws_session_token=credentials['SessionToken'],
                region_name=settings.AWS_REGION
            )
        else:
            # Fallback to default credentials if no role is specified
            self.s3_client = boto3.client('s3', region_name=settings.AWS_REGION)
            
        self.scaler = StandardScaler()
        
    def _load_data_from_s3(self, s3_path: str) -> pd.DataFrame:
        """Load data from S3 bucket."""
        bucket = s3_path.split('/')[0]
        key = '/'.join(s3_path.split('/')[1:])
        
        response = self.s3_client.get_object(Bucket=bucket, Key=key)
        df = pd.read_csv(response['Body'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
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
    
    def _create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create time series features."""
        df = data.copy()
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['dayofweek'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        df['lag_1'] = df['value'].shift(1)
        df['lag_7'] = df['value'].shift(7)
        df['lag_30'] = df['value'].shift(30)
        df['rolling_mean_7'] = df['value'].rolling(window=7).mean()
        df['rolling_std_7'] = df['value'].rolling(window=7).std()
        return df.dropna()
    
    def _find_best_arima_params(self, data: pd.Series) -> Tuple[int, int, int]:
        """Find best ARIMA parameters using AIC."""
        best_aic = float('inf')
        best_params = (1, 1, 1)
        
        for p in range(0, 3):
            for d in range(0, 2):
                for q in range(0, 3):
                    try:
                        model = sm.tsa.ARIMA(data, order=(p, d, q))
                        results = model.fit()
                        if results.aic < best_aic:
                            best_aic = results.aic
                            best_params = (p, d, q)
                    except:
                        continue
        return best_params
    
    def _train_arima(self, data: pd.DataFrame) -> Dict:
        """Train ARIMA model with optimized parameters."""
        best_params = self._find_best_arima_params(data['value'])
        model = sm.tsa.ARIMA(data['value'], order=best_params)
        results = model.fit()
        return {'model': results, 'params': best_params}
    
    def _train_sarima(self, data: pd.DataFrame) -> Dict:
        """Train SARIMA model for seasonal data."""
        model = SARIMAX(data['value'],
                       order=(1, 1, 1),
                       seasonal_order=(1, 1, 1, 12))  # Assuming monthly seasonality
        results = model.fit()
        return {'model': results}
    
    def _train_random_forest(self, data: pd.DataFrame) -> Dict:
        """Train Random Forest model with engineered features."""
        df = self._create_features(data)
        X = df.drop(['value'], axis=1)
        y = df['value']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_scaled, y)
        return {'model': model, 'scaler': self.scaler, 'feature_names': X.columns}
    
    def _train_linear_regression(self, data: pd.DataFrame) -> Dict:
        """Train Linear Regression model with engineered features."""
        df = self._create_features(data)
        X = df.drop(['value'], axis=1)
        y = df['value']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        model = LinearRegression()
        model.fit(X_scaled, y)
        return {'model': model, 'scaler': self.scaler, 'feature_names': X.columns}
    
    def _generate_arima_forecast(self, model_dict: Dict, steps: int) -> Dict:
        """Generate forecast using ARIMA model."""
        model = model_dict['model']
        forecast = model.forecast(steps=steps)
        conf_int = model.get_forecast(steps=steps).conf_int()
        
        return {
            'point_forecast': forecast.tolist(),
            'p30': conf_int.iloc[:, 0].tolist(),
            'p70': conf_int.iloc[:, 1].tolist()
        }
    
    def _generate_sarima_forecast(self, model_dict: Dict, steps: int) -> Dict:
        """Generate forecast using SARIMA model."""
        model = model_dict['model']
        forecast = model.forecast(steps=steps)
        conf_int = model.get_forecast(steps=steps).conf_int()
        
        return {
            'point_forecast': forecast.tolist(),
            'p30': conf_int.iloc[:, 0].tolist(),
            'p70': conf_int.iloc[:, 1].tolist()
        }
    
    def _generate_ml_forecast(self, model_dict: Dict, data: pd.DataFrame, steps: int) -> Dict:
        """Generate forecast using machine learning models."""
        model = model_dict['model']
        scaler = model_dict['scaler']
        feature_names = model_dict['feature_names']
        
        # Create future dates
        last_date = data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), periods=steps, freq='D')
        
        # Create features for future dates
        future_df = pd.DataFrame(index=future_dates)
        future_df = self._create_features(future_df)
        
        # Scale features
        X_future = future_df[feature_names]
        X_future_scaled = scaler.transform(X_future)
        
        # Generate predictions
        predictions = model.predict(X_future_scaled)
        
        # Calculate prediction intervals (simple approach)
        std_dev = np.std(model.predict(X_future_scaled))
        
        return {
            'point_forecast': predictions.tolist(),
            'p30': (predictions - 1.28 * std_dev).tolist(),
            'p70': (predictions + 1.28 * std_dev).tolist()
        }
    
    async def generate_forecast(
        self,
        s3_path: str,
        forecast_type: str,
        error_metric: str,
        sku: Optional[str] = None,
        steps: int = settings.DEFAULT_FORECAST_HORIZON
    ) -> Dict:
        """
        Generate forecast using specified model type.
        """
        try:
            # Load and preprocess data
            df = await self._load_data(s3_path, sku)
            train_data, test_data = self._split_data(df)
            
            # Train model and generate forecast
            start_time = time.time()
            if forecast_type == "arima":
                model, forecast, metrics = self._train_arima(train_data, test_data, error_metric)
            elif forecast_type == "lightgbm":
                model, forecast, metrics = self._train_lightgbm(train_data, test_data, error_metric)
            else:  # linear_regression
                model, forecast, metrics = self._train_linear_regression(train_data, test_data, error_metric)
            
            training_time = time.time() - start_time
            
            # Generate quantiles
            quantiles = self._generate_quantiles(forecast, model, train_data)
            
            # Prepare response
            forecast_points = self._prepare_forecast_points(forecast, quantiles)
            
            return {
                "forecast_points": forecast_points,
                "metrics": {
                    "model_name": forecast_type,
                    "error_metric": error_metric,
                    "error_value": metrics[error_metric],
                    "training_time": training_time,
                    "prediction_time": metrics.get("prediction_time", 0)
                },
                "best_model": forecast_type,
                "quantiles": quantiles,
                "created_at": datetime.utcnow()
            }
            
        except Exception as e:
            logging.error(f"Error in generate_forecast: {str(e)}")
            raise

    async def _load_data(self, s3_path: str, sku: Optional[str] = None) -> pd.DataFrame:
        """Load data from S3 and preprocess."""
        try:
            response = self.s3_client.get_object(
                Bucket=settings.S3_BUCKET,
                Key=s3_path
            )
            df = pd.read_csv(response['Body'])
            
            # Basic preprocessing
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            if sku:
                df = df[df['sku'] == sku]
            
            return df
            
        except Exception as e:
            logging.error(f"Error loading data from S3: {str(e)}")
            raise

    def _split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Split data into train and test sets."""
        split_idx = int(len(df) * (1 - settings.TRAIN_TEST_SPLIT))
        return df[:split_idx], df[split_idx:]

    def _train_arima(self, train_data: pd.DataFrame, test_data: pd.DataFrame, error_metric: str) -> Tuple[sm.tsa.arima.model.ARIMA, np.ndarray, Dict]:
        """Train ARIMA model with hyperparameter tuning."""
        def objective(trial):
            p = trial.suggest_int('p', 0, 5)
            d = trial.suggest_int('d', 0, 2)
            q = trial.suggest_int('q', 0, 5)
            
            model = sm.tsa.arima.model.ARIMA(train_data['value'], order=(p, d, q))
            results = model.fit()
            forecast = results.forecast(steps=len(test_data))
            
            error = self._calculate_error(test_data['value'], forecast, error_metric)
            return error

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=settings.N_TRIALS, timeout=settings.TIMEOUT)
        
        best_params = study.best_params
        model = sm.tsa.arima.model.ARIMA(train_data['value'], order=(best_params['p'], best_params['d'], best_params['q']))
        results = model.fit()
        
        forecast = results.forecast(steps=len(test_data))
        metrics = self._calculate_metrics(test_data['value'], forecast, error_metric)
        
        return model, forecast, metrics

    def _train_lightgbm(self, train_data: pd.DataFrame, test_data: pd.DataFrame, error_metric: str) -> Tuple[lgb.LGBMRegressor, np.ndarray, Dict]:
        """Train LightGBM model with hyperparameter tuning."""
        def objective(trial):
            params = {
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 100)
            }
            
            model = lgb.LGBMRegressor(**params)
            X_train = self._create_features(train_data)
            y_train = train_data['value']
            
            model.fit(X_train, y_train)
            X_test = self._create_features(test_data)
            forecast = model.predict(X_test)
            
            error = self._calculate_error(test_data['value'], forecast, error_metric)
            return error

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=settings.N_TRIALS, timeout=settings.TIMEOUT)
        
        best_params = study.best_params
        model = lgb.LGBMRegressor(**best_params)
        X_train = self._create_features(train_data)
        y_train = train_data['value']
        
        model.fit(X_train, y_train)
        X_test = self._create_features(test_data)
        forecast = model.predict(X_test)
        
        metrics = self._calculate_metrics(test_data['value'], forecast, error_metric)
        
        return model, forecast, metrics

    def _train_linear_regression(self, train_data: pd.DataFrame, test_data: pd.DataFrame, error_metric: str) -> Tuple[LinearRegression, np.ndarray, Dict]:
        """Train Linear Regression model with feature engineering."""
        X_train = self._create_features(train_data)
        y_train = train_data['value']
        X_test = self._create_features(test_data)
        
        model = LinearRegression()
        model.fit(X_train, y_train)
        
        forecast = model.predict(X_test)
        metrics = self._calculate_metrics(test_data['value'], forecast, error_metric)
        
        return model, forecast, metrics

    def _calculate_error(self, actual: np.ndarray, predicted: np.ndarray, metric: str) -> float:
        """Calculate specific error metric."""
        metrics = self._calculate_metrics(actual, predicted, metric)
        return metrics[metric]

    def _generate_quantiles(self, forecast: np.ndarray, model, train_data: pd.DataFrame) -> Dict:
        """Generate quantile forecasts."""
        # Simple implementation - can be enhanced based on model type
        std = np.std(train_data['value'])
        return {
            'P30': forecast - 0.524 * std,
            'P50': forecast,
            'P80': forecast + 0.842 * std,
            'P90': forecast + 1.282 * std,
            'P95': forecast + 1.645 * std
        }

    def _prepare_forecast_points(self, forecast: np.ndarray, quantiles: Dict) -> List[Dict]:
        """Prepare forecast points with confidence intervals."""
        points = []
        for i, value in enumerate(forecast):
            points.append({
                'timestamp': datetime.utcnow() + timedelta(days=i),
                'value': float(value),
                'lower_bound': float(quantiles['P30'][i]),
                'upper_bound': float(quantiles['P95'][i])
            })
        return points 
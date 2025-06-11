# BYODForecast

A scalable time series forecasting system that allows users to upload their data and get forecasts using various models.

## Features

- Upload CSV files from S3
- Choose between fast results or precise forecasts
- Multiple error metrics to optimize for
- Automated model selection and hyperparameter tuning
- Ensemble forecasting capabilities
- Interactive visualization of results

## Project Structure

```
BYODForecast/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── models/
│   │   └── services/
│   └── tests/
├── frontend/
│   ├── public/
│   └── src/
└── infrastructure/
```

## Setup Instructions

1. Clone the repository
2. Set up Python virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   Create a `.env` file in the root directory with:
   ```
   AWS_ACCESS_KEY_ID=your_access_key
   AWS_SECRET_ACCESS_KEY=your_secret_key
   AWS_REGION=your_region
   ```

4. Start the backend server:
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

5. Start the frontend development server:
   ```bash
   cd frontend
   npm install
   npm start
   ```

## Available Error Metrics

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)
- Symmetric Mean Absolute Percentage Error (SMAPE)

## Models

### Fast Results
- ARIMA
- Exponential Smoothing
- Prophet
- Linear Regression
- Random Forest

### Precise Results (Additional)
- DeepAR
- Temporal Fusion Transformer (TFT)

## License

MIT 
# BYODForecast API

The backend API for BYODForecast, built with FastAPI and Python.

## Features

- RESTful API endpoints
- Multiple forecasting models
- Automated model selection
- Error metric optimization
- S3 integration
- Static file serving

## Setup

1. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
Create a `.env` file in the root directory with:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
```

4. Start the server:
```bash
uvicorn app.main:app --reload
```

## API Endpoints

- `GET /` - Serve frontend application
- `POST /api/forecast` - Create new forecast
- `GET /api/forecast/{forecast_id}` - Get forecast results
- `GET /api/metrics` - Get available error metrics

## Development

- `uvicorn app.main:app --reload` - Start development server
- `pytest` - Run tests
- `black .` - Format code
- `flake8` - Lint code 
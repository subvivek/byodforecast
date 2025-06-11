# BYODForecast

A scalable time series forecasting system with a modern web interface.

## Project Structure

```
BYODForecast/
├── byodforecast-ui/     # Frontend React application
└── byodforecast-api/    # Backend FastAPI application
```

## Frontend (byodforecast-ui)

The frontend is built with React and Material-UI. See [byodforecast-ui/README.md](byodforecast-ui/README.md) for setup instructions.

## Backend (byodforecast-api)

The backend is built with FastAPI and Python. See [byodforecast-api/README.md](byodforecast-api/README.md) for setup instructions.

## Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BYODForecast.git
cd BYODForecast
```

2. Set up the frontend:
```bash
cd byodforecast-ui
npm install
npm start
```

3. Set up the backend:
```bash
cd ../byodforecast-api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

## Environment Variables

### Frontend
Create `byodforecast-ui/.env`:
```
REACT_APP_API_URL=http://localhost:8000
```

### Backend
Create `byodforecast-api/.env`:
```
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
AWS_REGION=your_region
```

## License

MIT 
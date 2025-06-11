# BYODForecast UI

The frontend application for BYODForecast, built with React and Material-UI.

## Features

- Modern, responsive UI
- S3 file path input
- Forecast type selection (Fast/Precise)
- Error metric selection
- SKU-specific forecasting
- Interactive visualizations

## Setup

1. Install dependencies:
```bash
npm install
```

2. Start development server:
```bash
npm start
```

3. Build for production:
```bash
npm run build
```

## Environment Variables

Create a `.env` file in the root directory with:
```
REACT_APP_API_URL=http://your-api-url
```

## Development

- `npm start` - Start development server
- `npm test` - Run tests
- `npm run build` - Build for production
- `npm run deploy` - Build and deploy to backend static directory 
import React, { useState } from 'react';
import {
  Container,
  Box,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Paper,
  Grid,
} from '@mui/material';
import axios from 'axios';

// Get the API base URL from environment or use relative path
const API_BASE_URL = process.env.REACT_APP_API_URL || '/api';

function App() {
  const [s3Path, setS3Path] = useState('');
  const [forecastType, setForecastType] = useState('fast');
  const [errorMetric, setErrorMetric] = useState('MAE');
  const [sku, setSku] = useState('');
  const [forecastId, setForecastId] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const response = await axios.post(`${API_BASE_URL}/forecast`, {
        s3_path: s3Path,
        forecast_type: forecastType,
        error_metric: errorMetric,
        sku: sku || undefined,
      });

      setForecastId(response.data.forecast_id);
    } catch (err) {
      setError(err.response?.data?.message || 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Container maxWidth="md">
      <Box sx={{ my: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          BYODForecast
        </Typography>
        
        <Paper sx={{ p: 3 }}>
          <form onSubmit={handleSubmit}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="S3 Path"
                  value={s3Path}
                  onChange={(e) => setS3Path(e.target.value)}
                  required
                  helperText="Enter the S3 path to your CSV file"
                />
              </Grid>

              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Forecast Type</InputLabel>
                  <Select
                    value={forecastType}
                    label="Forecast Type"
                    onChange={(e) => setForecastType(e.target.value)}
                  >
                    <MenuItem value="fast">Fast Results</MenuItem>
                    <MenuItem value="precise">Most Precise Forecast</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Error Metric</InputLabel>
                  <Select
                    value={errorMetric}
                    label="Error Metric"
                    onChange={(e) => setErrorMetric(e.target.value)}
                  >
                    <MenuItem value="MAE">Mean Absolute Error (MAE)</MenuItem>
                    <MenuItem value="RMSE">Root Mean Squared Error (RMSE)</MenuItem>
                    <MenuItem value="MAPE">Mean Absolute Percentage Error (MAPE)</MenuItem>
                    <MenuItem value="SMAPE">Symmetric Mean Absolute Percentage Error (SMAPE)</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="SKU (Optional)"
                  value={sku}
                  onChange={(e) => setSku(e.target.value)}
                  helperText="Enter SKU to view specific forecast"
                />
              </Grid>

              <Grid item xs={12}>
                <Button
                  type="submit"
                  variant="contained"
                  color="primary"
                  fullWidth
                  disabled={loading}
                >
                  {loading ? 'Processing...' : 'Generate Forecast'}
                </Button>
              </Grid>
            </Grid>
          </form>

          {error && (
            <Typography color="error" sx={{ mt: 2 }}>
              {error}
            </Typography>
          )}

          {forecastId && (
            <Typography sx={{ mt: 2 }}>
              Forecast ID: {forecastId}
            </Typography>
          )}
        </Paper>
      </Box>
    </Container>
  );
}

export default App; 
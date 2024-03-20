import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Step 1: Load and preprocess data( dealing with missing value, format changing)
data = pd.read_csv("investment_daily_flows.csv") 
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Step 2: Visualize data
plt.figure(figsize=(10, 6))
plt.plot(data)
plt.title('Investment Daily Flows')
plt.xlabel('Date')
plt.ylabel('Flows')
plt.show()

# Step 3: SARIMA model selection
# Plot ACF and PACF to determine SARIMA parameters
plot_acf(data)
plot_pacf(data)
plt.show()


# use the auto_arima function from the pmdarima library
# Step 33: SARIMA model selection using auto_arima
# Use auto_arima to select the best SARIMA parameters
model = auto_arima(data, seasonal=True, m=7)  # Set m=7 for weekly seasonality
print(model.summary())

# Step 44: Fit SARIMA model
# Fit SARIMA model with the selected parameters
model.fit(data)

# Step 55: Model evaluation
# Forecast
forecast = model.predict(n_periods=len(data))

# # Calculate MAE and RMSE
# mae = np.mean(np.abs(forecast - data.values))
# rmse = np.sqrt(np.mean((forecast - data.values)**2))

# print(f"Mean Absolute Error (MAE): {mae}")
# print(f"Root Mean Squared Error (RMSE): {rmse}")


# Step 4: Fit SARIMA model
# Example parameters (replace with your own)
p, d, q = 2, 3, 1
# p, d, q = 1, 1, 1
P, D, Q, s = 1, 1, 1, 7  # Seasonal parameters (replace with appropriate values)

# Fit SARIMA model
model = SARIMAX(data, order=(p, d, q), seasonal_order=(P, D, Q, s))
results = model.fit()

# Step 5: Model evaluation
# Validate model using train-test split
train_size = int(len(data) * 0.8)
train_data, test_data = data[:train_size], data[train_size:]

# Forecast
forecast = results.get_forecast(steps=len(test_data))
forecast_values = forecast.predicted_mean

# Calculate MAE and RMSE
mae = np.mean(np.abs(forecast_values - test_data.values))
rmse = np.sqrt(np.mean((forecast_values - test_data.values)**2))

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Step 6: Forecasting and interpretation
plt.figure(figsize=(10, 6))
plt.plot(data, label='Original data')
plt.plot(test_data.index, forecast_values, label='Forecast', color='red')
plt.fill_between(test_data.index, forecast.conf_int()[:, 0], forecast.conf_int()[:, 1], color='pink', alpha=0.3)
plt.title('Investment Daily Flows Forecast')
plt.xlabel('Date')
plt.ylabel('Flows')
plt.legend()
plt.show()



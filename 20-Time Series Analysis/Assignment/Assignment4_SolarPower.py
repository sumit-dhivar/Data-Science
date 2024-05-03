# -*- coding: utf-8 -*-
"""
Created on Wed March 28 14:47:53 2024

@author: sumit
"""

# Business Objective :

# Maximize : The use of solar power so that the natural things may not get harm
    
# Minimize : The use of non-rennewable source
    
# Business contraints : Availability of solar systms and budget 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('solarpower.csv')

df.columns

# Check for missing values
print(df.isnull().sum())
# There is no null value is present 

# Summary statistics
print(df.describe())    
# There is  slight more difference in mean and median . So may be the data is not stationary
# lets check
# Check for the outlierss
sns.boxplot(df)
# There is no outliers
# Plot the distplot to checkwhether data is normally distributed or not
sns.distplot(df.cum_power)
# Data is smowhat bi-modal in nature

# Plotting time series
df['cum_power'].plot(figsize=(10, 6))
plt.ylabel('Cumulative Power')
plt.title('Solar Power Consumption Over Time')
plt.show()
# the data shows increases so the data is not stationary

# Convert 'date' column to datetime format
df['date'] = pd.to_datetime(df['date'])

# Set the 'date' column as the index
df.set_index('date', inplace=True)

# Now, let's try resampling again
df_monthly = df.resample('M').mean().fillna(method='ffill')

# Time Series Decomposition
from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(df_monthly, model='additive')
decomposition.plot()
plt.show()

# By seasonal decomposition The data become more constant and stationary

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ACF and PACF plots
plot_acf(df_monthly)
plot_pacf(df_monthly)
plt.show()

# Define and fit the ARIMA model
# Note: You might need to adjust p, d, q values based on ACF and PACF plots
model = ARIMA(df_monthly, order=(1,1,1)) # These are example parameters
results = model.fit()

# Summary of the model
print(results.summary())

# Forecasting Future Solar Power Consumption
# Forecasting the next 12 months as an example
forecast = results.forecast(steps=12)

# Plotting the forecast
plt.figure(figsize=(10, 6))
plt.plot(df_monthly.index, df_monthly['cum_power'], label='Historical Monthly Mean')
plt.plot(forecast.index, forecast, label='Forecast')
plt.legend()
plt.show()

# as we see the solor power consumptions are increasing on consistently in future

from statsmodels.tsa.arima.model import ARIMA
import pmdarima as pm

# Auto ARIMA to find best parameters
model = pm.auto_arima(df['cum_power'], seasonal=True, m=12, trace=True)

# Fit ARIMA model
model_fit = ARIMA(df['cum_power'], order=model.order).fit()

# Summary of the model
print(model_fit.summary())

# Forecast
forecast = model_fit.forecast(steps=12)  # For example, next 12 periods
print(forecast)

# Plotting the forecast
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['cum_power'], label='Historical Daily Price')
plt.plot(forecast.index, forecast, label='Forecasted Price')
plt.legend()
plt.show()













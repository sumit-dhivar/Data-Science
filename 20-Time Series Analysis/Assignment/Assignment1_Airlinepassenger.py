# -*- coding: utf-8 -*-
"""
Created on Wed March 28 15:33:49 2024

@author: sumit
"""
"""
Business Objective :

Maximize : Maximize the number of passengers travels from the airlines
    
Minimize : The total cost so that the number of passengers may increases in future
    
Business Contraints : The data security as well as the prediction rate or any uncertain event goes happen in future which will affect on the prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
from datetime import datetime
import warnings
warnings.filterwarnings('ignore') 

data=pd.read_excel('Airlines Data.xlsx')
data.shape 

# Creating the 'Date' as Index for data and viewing the dataset
data['Month']=pd.to_datetime(data['Month'], infer_datetime_format=True)
data=data.set_index(['Month'])
print(data.head())
print(data.tail())

# Visualizing the Time Series plot for the number of Air Passengers
plt.figure(figsize=(20,10))
plt.xlabel("Month")
plt.ylabel("Number of Air Passengers")
plt.plot(data)
# The nature of the data is in increasing pattern

# As we know that, A stationary Time Series is one which characteristics like mean and variance does not change over time
# Preparing the data for analyzing the Rolling Mean and Standard Deviation to check stationarity for given Time Series
rolmean=data.rolling(window=12).mean()
rolstd=data.rolling(window=12).std()
print(rolmean.head(15))
print(rolstd.head(15))

# Plotting the Rolling Mean and Standard Deviation, which has window of 12
plt.figure(figsize=(20,10))
actual=plt.plot(data, color='red', label='Actual')
mean_6=plt.plot(rolmean, color='green', label='Rolling Mean') 
std_6=plt.plot(rolstd, color='black', label='Rolling Std')
plt.legend(loc='best')
plt.title('Rolling Mean & Standard Deviation')
plt.show(block=False)
# By looking below plot, we conclude that, it is non-stationary bcoz mean and variance is not constant

# Using ADF (Augmented Dickey-Fuller Test) to check stationarity, another approach based on statistics.
# ADF has : Null hypothesis - Time Series is non-stationary
from statsmodels.tsa.stattools import adfuller
print('Dickey-Fuller Test: ')
dftest=adfuller(data['Passengers'], autolag='AIC')
dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','No. of Obs'])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)
# From above ADF test, we fail to reject the null hypothesis, since p-value is greater than 0.05
# Below we took log transformation to make our Time series stationary and plotted visual for it
# We found graph upward trending over time with seasonality

plt.figure(figsize=(20,10))
data_log=np.log(data)
plt.plot(data_log)

# Testing Rolling Mean with window 12 on above log transformation and concluded non-stationary, again
plt.figure(figsize=(20,10))
MAvg=data_log.rolling(window=12).mean()
MStd=data_log.rolling(window=12).std()
plt.plot(data_log)
plt.plot(MAvg, color='blue')

# Use nother method
data_log_diff=data_log-MAvg
data_log_diff.head(12)

data_log_diff=data_log_diff.dropna()
data_log_diff.head()

def stationarity(timeseries):
    
    rolmean=timeseries.rolling(window=12).mean()
    rolstd=timeseries.rolling(window=12).std()
    
    plt.figure(figsize=(20,10))
    actual=plt.plot(timeseries, color='red', label='Actual')
    mean_6=plt.plot(rolmean, color='green', label='Rolling Mean') 
    std_6=plt.plot(rolstd, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    print('Dickey-Fuller Test: ')
    dftest=adfuller(timeseries['Passengers'], autolag='AIC')
    dfoutput=pd.Series(dftest[0:4], index=['Test Statistic','p-value','Lags Used','No. of Obs'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)

# Checking the stationarity of differenced data
# From Rolling method, we see that Mean and standard deviation is not varying.
# From ADF, we reject the null hypothesis bcoz p-value is less than 0.05 (significance level)
# Applying all the transformation and methods, our differenced data is now stationary

stationarity(data_log_diff)

# Checking for Trend stationarity and performing exponential transformation on our data
plt.figure(figsize=(20,10))
exp_data=data_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
plt.plot(data_log)
plt.plot(exp_data, color='black')
# Since log transformation is non-stationary, applying differencing on our data.

exp_data_diff=data_log-exp_data
stationarity(exp_data_diff)
# From above graph and ADF test, we conclude, it is trend stationary

plt.figure(figsize=(20,10))
data_shift=data_log-data_log.shift()
plt.plot(data_shift)

data_shift=data_shift.dropna()
stationarity(data_shift)

# Decomposing Time Series into its components : Trend, Seasonality and Residual
from statsmodels.tsa.seasonal import seasonal_decompose
decomp=seasonal_decompose(data_log)

trend=decomp.trend
seasonal=decomp.seasonal
residual=decomp.resid

plt.subplot(411)
plt.plot(data_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()

# Plotting the ACF and PACF to find q and p value
# We got q and p both 2 from the graph
from statsmodels.tsa.stattools import acf, pacf

lag_acf=acf(data_shift, nlags=20)
lag_pacf=pacf(data_shift, nlags=20, method='ols')

plt.figure(figsize=(20,10))
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='green')
plt.axhline(y=-1.96/np.sqrt(len(data_shift)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(data_shift)),linestyle='--',color='green')
plt.title('Autocorrelation Function')

plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='green')
plt.axhline(y=-1.96/np.sqrt(len(data_shift)),linestyle='--',color='green')
plt.axhline(y=1.96/np.sqrt(len(data_shift)),linestyle='--',color='green')
plt.title('Partial Autocorrelation Function')

# Building the ARIMA model with p=2, q=2 and I=1
from statsmodels.tsa.arima_model import ARIMA
import statsmodels.api as sm

plt.figure(figsize=(20, 10))

# Assuming data_log is your time series data
model = sm.tsa.arima.ARIMA(data_log, order=(2, 1, 2))  # Use ARIMA from statsmodels.tsa.arima.model
results = model.fit()
plt.plot(data_shift)  # Assuming data_shift is plotted here
plt.show()

# Predicted values from ARIMA model in difference form
predictions=pd.Series(results.fittedvalues, copy=True)
print(predictions.head())

# Performing Inverse Transformation for differencing, by doing cumulative sum
predictions_cum_sum=predictions.cumsum()
print(predictions_cum_sum.head())

# Below is data display of predicted value in log scale
predictions_log = pd.Series(data_log['Passengers'].iloc[0], index=data_log.index)
predictions_log = predictions_log.add(predictions_cum_sum, fill_value=0)
predictions_log.head()

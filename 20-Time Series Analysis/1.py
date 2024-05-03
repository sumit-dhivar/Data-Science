# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 15:08:52 2024

@author: sumit

"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
plt.style.use('dark_background')

#load data set 
df = pd.read_csv('AirPassengers.csv')
df.columns 
df.rename({'#Passengers':'Passenger'},axis=1,inplace=True)

print(df.dtypes)
#Month is text and passengers in int 
#Now let us convert into date and time 
df['Month'] = pd.to_datetime(df['Month']) 
print(df.dtypes)

df.set_index('Month' , inplace = True )

plt.plot(df.Passenger)
#There is increasing trend and it has got seasonality 
#Is the data stationary? 
#------------------------Dickey - Fuller Test---------------------------
from statsmodels.tsa.stattools import adfuller 
adf, pvalue, usedlag_, noobs_, critical_values_, icbest_ = adfuller(df)
print("pvalue", pvalue, "if above 0.05, data is not stationary ")
#Since data is not stationary, we may need SARIMA(modern model) and not just ARIMA 
#Now let us extract the year and month form the date and time column 
df['year'] = [d.year for d in df.index]
df['month'] = [d.strftime("%b") for d in df.index]
years = df['year'].unique()

#Plot yearly and monthly values as boxplot 
sns.boxplot(x = 'year', y = 'Passenger', data = df)
#No. of passengers are going up year by year 
sns.boxplot(x='month', y='Passenger', data = df)
#Owner all there is higher trend in July and August compared to rest of the 

#Extract and plot trend, seasonal and residuals. 
from statsmodels.tsa.seasonal import seasonal_decompose
decomposed = seasonal_decompose(df['Passenger'], model='additive')
#Additive time series 
#value = Basic Level + Trend + Seasonality + Error 


trend = decomposed.trend
seasonal = decomposed.seasonal

residual = decomposed.resid

plt.figure(figsize=(12,8))
plt.subplot(411)
plt.plot(df['Passenger'], label = 'Original', color = 'yellow')
plt.legend(loc='upper left')
plt.subplot(412)
plt.plot(trend, label = 'Trend', color = 'yellow')
plt.legend(loc='upper left')
plt.subplot(413)
plt.plot(seasonal, label = 'Seasonal', color = 'yellow')
plt.legend(loc='upper left')
plt.subplot(414)
plt.plot(residual, label = 'Residual', color = 'yellow')
plt.legend(loc='upper left')
plt.show()

'''
Trend is going up from 1950s to 60s 
It is highly seasonal showing peaks at particular interval 
This helps to select specific prediction model
'''
#AUTOCORRELATION
#values are not correlated wiht x-axis nbut with its lag 
#meaning yesturdays value is depend on day before yesterday so on so forth 
#Autocorrection is simply the correlation of a series with its own lagss
#Plot lag on x axis and correlation on y axis 
#Any correlation above confidence lines are statistically significant 

from statsmodels.tsa.stattools import acf 
acf_144 = acf(df.Passenger, nlags=144)
plt.plot(acf_144)
#Auto correlation above zero means positive correlation and below as negative 
#Obtain the same but with single ine and more info... 
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(df.Passenger)
#Any lag before 40 has positive correlatino
#horizontal bands indicate 95% and 99%(dashed) confidence bands 





















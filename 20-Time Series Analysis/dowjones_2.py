# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 16:02:09 2024

@author: sumit

ARMA
"""

import numpy as np 
import pandas as pd 
import seaborn as sns

sns.set_theme()

dowjones = sns.load_dataset('dowjones')
dowjones.head()
sns.lineplot(data=dowjones, x='Date',y='Price')

"""
A simple moving average (SMA) calculates the average of a selected range of values,
by the number of periods in that range.
The most typical moving averages are 30-day, 50-day, 100-day and 365 day moving 
averages. Moving averages are nice cause they can determine trends while ignoring
short-term fluctuations.
One can calculate the sma by simply using 

"""
dowjones['sma_30'] = dowjones['Price'].rolling(window=30,min_periods=1).mean()
dowjones['sma_50'] = dowjones['Price'].rolling(window=50,min_periods=1).mean()
dowjones['sma_100'] = dowjones['Price'].rolling(window=100,min_periods=1).mean()
dowjones['sma_365'] = dowjones['Price'].rolling(window=365,min_periods=1).mean()

sns.lineplot(x='Date', y='value', legend='auto', hue='variable', data=dowjones.melt('Date'))
'''
As you can see the higher the value of the window, 
the lesser it is affected by short-term fluctuations 
and it captures long-term trends in the data. 
Simple Mobing averages are often used by traders 
in the stock market for technical analysis 

'''
##########Exponential Moving Average
 
'''
Simple moving averages are nice, but 
they give equal weightage to each of the data points, 
what if you wanted an average that will give higher weight 
to more recent points and lessr to points in the past. In that cae, 
what yoou want is to compute the exponential moving average (EMA)
'''

dowjones['ema_50'] = dowjones['Price'].ewm(span=50, adjust=False).mean()
dowjones['ema_100'] = dowjones['Price'].ewm(span=100, adjust=False).mean()

sns.lineplot(x='Date',y='value', legend='auto', hue='variable', data=dowjones[['Date','ema_50','sma_50']].melt('Date'))

'''
As you can see the ema_50 follows the Price chart more closely 
than the sma_50 and is more sensitive to the recent data points.

Which moving average you should use as a feature of your forecasting model

'''



























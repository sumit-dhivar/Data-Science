# -*- coding: utf-8 -*-
"""
Created on Wed March 28 14:43:16 2024

@author: sumit
"""

# Business Objective :

# Maximize : The sales of the plastic in the future so that the overall revenue of the company be increased
    
# Minimize : The wastage of the plastic and the production cost
    
# Business Contraints : The availability of material and Budget of the company

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
import statsmodels.graphics.tsaplots as tsa_plots
import statsmodels.tsa.statespace as tm_models
from datetime import datetime,time

plastic = pd.read_csv("PlasticSales.csv")

plastic.head()

# Converting the normal index of Amtrak to time stamp 
plastic.index = pd.to_datetime(plastic.Month,format="%b-%y")

plastic.columns

sns.boxplot(plastic)
# No outlier is present in the data

plastic.info()
# There is no null value present in the data 
# Month is string and Sales is numeric in nature

# Time series plot
plastic["Sales"].plot()
# As we observed the data is not stationary

# Creating a Date column to store the actual Date format for the given Month column
plastic["Date"] = pd.to_datetime(plastic.Month,format="%b-%y")

# Extracting Day, weekday name, month name, year from the Date column using 
# Date functions from pandas 

plastic["month"] = plastic.Date.dt.strftime("%b") # month extraction
plastic["year"] =plastic.Date.dt.strftime("%Y") # year extraction

# Heat map visualization 
heatmap_y_month = pd.pivot_table(data=plastic,values="Sales",index="year",columns="month",aggfunc="mean",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g")

# Boxplot for ever
sns.boxplot(x="month",y="Sales",data=plastic)
sns.boxplot(x="year",y="Sales",data=plastic)

# Line plot for Ridership based on year  and for each month
sns.lineplot(x="year",y="Sales",hue="month",data=plastic)
# As we see in data there is some variation according to the month
# for jan,jun,july ,aug,sept the plastic sales suddenly increases

# moving average for the time series to understand better about the trend character in Plastic
plastic.Sales.plot(label="org")
for i in range(2,24,6):
    plastic["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc=3)

# As we see in the following output the sales constatntly increases 
# Means this is not stationary 

# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(plastic.Sales,model="additive")
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(plastic.Sales,model="multiplicative")
decompose_ts_mul.plot()

# As we see there is stillthe nature is non-stationary for additive
# But when we comes to multiplicative the data is become constant and statinary

# ACF plots and PACF plots on Original data sets 
tsa_plots.plot_acf(plastic.Sales,lags=10)
tsa_plots.plot_pacf(plastic.Sales)

# splitting the data into Train and Test data and considering the last 12 months data as 
# Test data and left over data as train data 

Train = plastic.head(48)
Test = plastic.tail(12)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org))*100/org
    return np.mean(temp)

# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales) # 17.04

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales) #101.985

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12,damped=True).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales) # 14.422

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales) #14.994













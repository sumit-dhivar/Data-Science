# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 09:00:37 2024

@author: sumit
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import Holt 
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#Now to load the datasets 
cocacola = pd.read_excel('CocaCola_Sales_Rawdata.xlsx')
#Let us plot the datasett and its nature 
cocacola.Sales.plot()
#Splittiing the data into train and test set data
#since we are working on quaterly datasets and in y 
#Test data = 4 and traiin data is 38 
Train = cocacola.head(38)
Test = cocacola.tail(4)

#here we are considering [erformance ] parameters as  
#rather thean mean square error 
#custom function is written to calculate MPSE 
def MAPE(pred,org):
    temp=np.abs((pred-org)/org)*100 
    return np.mean(temp)

#EDA which comprises identification of level, trends and 
#In order to seperate Trend and Seasonality moving average 
mv_pred = cocacola['Sales'].rolling(4).mean()
mv_pred.tail(4)
#Now let us calculate mean absolute percentage error of theses values 
MAPE(mv_pred.tail(4), Test.Sales)
#moving average is prediction complete values, out of which 
#aere considered as perdicted values and last four values 
#basic purpose of moving averagge is deseasonalizing 

cocacola.Sales.plot(label='org')
#This is original plot 
#Now let us seperate out Trend and Seasonality 
for i in range(2,9,2):
    #it will take window of 2,4,6,8 
    cocacola['Sales'].rolling(i).mean().plot(label=str(i))
    plt.legend(loc=3)
    
#you can see i=4 and i=8 aare deseasonavle plots 
#Time sseries decomposition is the another technizue of sep 
decompose_ts_add = seasonal_decompose(cocacola.Sales,model='additive', period=4)
print(decompose_ts_add.trend)
print(decompose_ts_add.seasonal)
print(decompose_ts_add.resid)
print(decompose_ts_add.observed)
decompose_ts_add.plot()

decompose_ts_mul = seasonal_decompose(cocacola.Sales,model='multiplicative', period=4)
print(decompose_ts_mul.trend)
print(decompose_ts_mul.seasonal)
print(decompose_ts_mul.resid)
print(decompose_ts_mul.observed)
decompose_ts_mul.plot()

#you can observe the difference between these plots 
#Now let us plot ACF plot to check the auto correlation 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales, lags=4)

#we can observe the output in which r1, r2, r3 and r4 has high 
#This is all about EDA 
#Let us apply data to data driven models 
#simple exponential method 
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
ses_model = SimpleExpSmoothing(Train['Sales']).fit()
pred_ses = ses_model.predict(start=Test.index[0], end=Test.index[-1])
#Now calculate MAPE 
MAPE(pred_ses, Test.Sales)
#we are getting 8.307 
#Holts exponential smoothng 
#here only trendd is captured 
hw_model = Holt(Train['Sales']).fit()
pred_hw = hw_model.predict(start=Test.index[0], end=Test.index[-1])
MAPE(pred_hw, Test.Sales)

#Holts winter exponential smoothing with additive seasonal 
hwe_model_add_add = ExponentialSmoothing(Train['Sales'],seasonal_periods=4).fit()
pred_hwe_model_add_add =hwe_model_add_add.predict(start=Test.index[0], end=Test.index[-1])
MAPE(pred_hwe_model_add_add,Test.Sales)
#Holts winter exponential smoothing with multiplicative seasonal 
hwe_model_mul_add = ExponentialSmoothing(Train['Sales'],seasonal_periods=4).fit()
pred_hwe_model_mul_add = hwe_model_mul_add.predict(start=Test.index[0], end=Test.index[-1])
MAPE(pred_hwe_model_mul_add,Test.Sales)

#let us apply to complet edata of cocacola 
#we have seen that hwe_model_add_add has lowest MAPE value 
hwe_model_add_add = ExponentialSmoothing(cocacola['Sales'],seasonal='add',seasonal_periods=4).fit()
#import the new datasets for which prediction has to  
new_data = pd.read_excel('CocaCola_Sales_New_Pred.xlsx')
newdata_pred = hwe_model_add_add.predict(start=new_data.index[0], end=new_data.index[-1])































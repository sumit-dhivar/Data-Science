# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:21:30 2024

@author: sumit

Time Series Prediction on Wallmart Dataset and Predict_new
"""

import pandas as pd 
import numpy as np

Walmart = pd.read_csv('Walmart Footfalls Raw.csv')

#pre-processing 
Walmart['t'] = np.arange(1,160)
Walmart['t_square'] = Walmart['t'] * Walmart['t']
Walmart['log_footfalls'] = np.log(Walmart['Footfalls'])
Walmart.columns
# ['Month', 'Footfalls', 't', 't_square', 'log_footfalls'] 
#In Walmart data we have Jan-1991 to 0th column, we need only first 
#example - Jan from each cell 

p = Walmart['Month'][0]
#before we will extract, let us create new column called 
#months to store extracted values 
p[0:3]

Walmart['months'] = 0
#you can check the dataframe with months name with all values 0 
#the total records are 159 in walmart 
for i in range(159):
    p = Walmart['Month'][i]
    Walmart['months'][i] = p[0:3] 
    
month_dummies = pd.DataFrame(pd.get_dummies(Walmart['months']))
#now let us concatenate these dummy values to dataframe 
Walmart1 = pd.concat([Walmart, month_dummies], axis=1)
#you can check the dataframe walmart1 

#Visualization - time Plot 
Walmart1.Footfalls.plot()

#Data Partition 
Train = Walmart1.head(147) 
Test = Walmart1.tail(12)

#to change the index value in pandas dataframe 
#Test.set_index(np.arange(1,13)) 

############----Linear------############
import statsmodels.formula.api as smf 

linear_model = smf.ols('Footfalls ~ t' , data = Train).fit()
pred_linear = pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))

rmse_linear = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_linear))**2))
rmse_linear
# 209.92559265462643
####################Exponential#############################
Exp = smf.ols('log_footfalls ~ t' , data = Train).fit()
pred_Exp= pd.Series(Exp.predict(pd.DataFrame(Test['t'])))

rmse_Exp = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Exp))**2))
rmse_Exp
# 2062.9501
################---Quadratic----#########################
Quad = smf.ols('Footfalls ~ t + t_square' , data = Train).fit()
pred_Quad= pd.Series(Quad.predict(pd.DataFrame(Test[['t','t_square']])))

rmse_Quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_Quad))**2))
rmse_Quad
# 137.154627
################################################################### 

#############Additive Seasonability ###############################3 
add_sea = smf.ols('Footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data = Train).fit()
pred_add_sea =  pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_add_sea))**2))
rmse_add_sea
# 264.6643
################Multiplicative Seasonality #################################### 

Mul_sea = smf.ols('log_footfalls ~ Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data = Train).fit()
pred_mul_sea =  pd.Series(Mul_sea.predict(Test))
rmse_mul_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_mul_sea))**2))
rmse_mul_sea
# 2062.9960
#############-------Additive Seasonality Quadratic Trend --------------- 
add_sea_Quad = smf.ols('Footfalls ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data = Train).fit()
add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_mul_sea))**2))
rmse_add_sea_quad
# 2062.996088663918
############---------Multiplicative Seasonality Linear Trend-------------###
Mul_add_sea = smf.ols('log_footfalls ~ t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data = Train).fit()
pred_mul_add_sea =  pd.Series(Mul_add_sea.predict(Test))
rmse_mul_add_sea = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_mul_add_sea))**2))
rmse_mul_add_sea
# 2062.9434993334708

#################-------------Consolidate-----------####################33 

data = {'MODEL':pd.Series(['rmse_linear','rmse_Exp','rmse_Quad','rmse_add_sea','rmse_mul_sea','rmse_add_sea_quad','rmse_mul_add_sea']),
        "RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_mul_sea,rmse_add_sea_quad,rmse_mul_add_sea])}
table_rmse = pd.DataFrame(data)
table_rmse

#################---------Testing ################## 

predict_data = pd.read_excel('Predict_new.xlsx')

model_full = smf.ols('Footfalls ~ t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov', data = Train).fit()
pred_model_full = pd.Series(add_sea.predict(Test[['Jan' , 'Feb' , 'Mar' , 'Apr' , 'May' , 'Jun' , 'Jul' , 'Aug' , 'Sep' , 'Oct' , 'Nov']]))
rmse_model_full = np.sqrt(np.mean((np.array(Test['Footfalls']) - np.array(pred_model_full))**2))
rmse_add_sea

pred_new = pd.Series(model_full.predict(predict_data))
pred_new

predict_data['forecasted_Footfalls'] = pd.Series(pred_new)

#Autoregression Model (AR)
#Calculating Residuals from best model applied on full data 
#AV - FV 
full_res =  Walmart1.Footfalls - model_full.predict(Walmart1)

#ACF plot on residuals 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(full_res,lags=12)
#ACF is an (complete) auto-correlation function gives values 
#of auto-correlatin of any time series with its lagged values 

#PACF is a partial auto-correlation function 
#It finds correlationn of present with lags of the residual of the  

tsa_plots.plot_pacf(full_res, lags=12)
#Alternative approach for ACF plot 
#from pandas.plotting import autocorrelation_plot 
#autocorrelation_ppyplot.show() 

#AR model 
from statsmodels.tsa.ar_model import AutoReg
model_ar = AutoReg(full_res, lags=[1])
#model_ar = AutoReg(Train_res, lags=12) 
model_fit = model_ar.fit()

print('Coefficients: %s' %model_fit.params)

pred_res = model_fit.predict(start=len(full_res), end=len(full_res)+len(predict_data)-1, dynamic=False)
pred_res.reset_index(drop=True, inplace=True) 

#The Final Predictions using ASQT and AR(1) Model 
final_pred = pred_new + pred_res 
final_pred














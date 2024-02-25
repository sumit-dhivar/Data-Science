# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 15:31:46 2024

@author: sumit
"""
import pandas as pd
import numpy as np
import seaborn as sns
wcat = pd.read_csv('wc-at.csv')
#EDA 
# 1. Measure the central tendancy 
# 2. Measures of dispersion
# 3. Third Moment Buisness Analysis
# 4. Fourth Moment Buisness decision

wcat.info
wcat.describe()
#Graphical Representation 
import matplotlib.pyplot as plt
plt.bar(height=wcat.AT, x = np.arange(1,110,1))
plt.hist(wcat.AT)
plt.boxplot(wcat.AT)
sns.distplot(wcat.AT)
#data is right skewed 
#scatter plot 
plt.scatter(x = wcat['Waist'],y=wcat['AT'],color='green')
#direction : Positive
#Linearity: Moderate
# Strength:poor 
#Now let's calculate correlation coefficient 
np.corrcoef(wcat.Waist,wcat.AT)
#moderately correlated 0.815<0.85
#Let is check the direction using covar factor 
cov_output = np.cov(wcat.Waist,wcat.AT)[0,1]
cov_output
#635.91 it is positive means correlation will be positive
#Now let us apply to linearr regression model 
import statsmodels.formula.api as smf
#All machine learning algorithms are implemented using sklearn 
#but for thsis statsmodels 
#package is beign used because it gives you  
#backend calculatioon of beta-0 and beta-1 
model=smf.ols('AT~Waist',data = wcat).fit()
model.summary()
#OLS helps to find best fit model, which causes 
#least square error 
#first you check R squared value  = 0.670, if R square = 0.8 means that 
# model is best fit 
#fit, it R-squared = 0.8 to 0.6 moderate fit.
#Next you check P>|t| = 0, it means less than alpha,
#alpha is 0.05, Hence the model is accepted 
#p=00<0.05 hence acceptable 
#beta-0 = -215.98 
#beta-1 = 3.45


#Regression Line 
pred1 = model.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(wcat.Waist,wcat.AT)
plt.plot(wcat.Waist,pred1,"r")
plt.show()

#Error Calculations 
res1 = wcat.AT-pred1
np.mean(res1)
#It must be zero and here it 10^-14=~0
res_sqr1 = res1*res1
mse1 = np.mean(res_sqr1)
rmse1 = np.sqrt(mse1)
rmse1
#32.76.76 lesser the value better the model 
#how to improve this model, transfromation of 
plt.scatter(x=np.log(wcat['Waist']), y=wcat['AT'], color='brown')
#Data is linearly scattered , direction: positive strength: poor
#Now let us check the correlation coefficient
np.corrcoef(np.log(wcat.Waist),wcat.AT)
#r value is 0.82 < 0.85 hence moderate linerarity 
model2 = smf.ols('AT~np.log(Waist)',data = wcat).fit()
#Y is AT and X = log(Waist)
model2.summary()

# OLS Regression Results                            
# ==============================================================================
# Dep. Variable:                     AT   R-squared:                       0.675
# Model:                            OLS   Adj. R-squared:                  0.672
# Method:                 Least Squares   F-statistic:                     222.6
# Date:                Wed, 21 Feb 2024   Prob (F-statistic):           6.80e-28
# Time:                        16:33:39   Log-Likelihood:                -534.11
# No. Observations:                 109   AIC:                             1072.
# Df Residuals:                     107   BIC:                             1078.
# Df Model:                           1                                         
# Covariance Type:            nonrobust     

#Again check the R-squarre value=0.67 which is less than 0.8 
#p vaue is 0 less than 0.05 
pred2  = model2.predict(pd.DataFrame(wcat['Waist']))
plt.scatter(np.log(wcat.Waist),wcat.AT)
plt.plot(np.log(wcat.Waist),pred2,"r")
plt.legend(['observed data','predicted line'])
plt.show()

#Error calculation  
res2 = wcat.AT-pred2
res_sqr2 = res2 * res2
mse2 = np.mean(res_sqr2)
rmse2 = np.sqrt(mse2)
rmse2
#32.496 
#Ther are no significant change as 0.821, RSquare = 0.675 and RMS
#There no considerable changes 
#Now let us change y value instead of x
plt.scatter(x=wcat['Waist'],y=np.log(wcat['AT']),color='orange')
np.corrcoef((wcat.Waist,np.log(wcat.AT)))
#r value is 0.84 <0.85 hence moderate linerarity 
model3 = smf.ols('np.log(AT)~Waist',data=wcat).fit()
model3.summary()
#Again check the R-squred value = 0.707 which is less than 0.8 
#p value is 0.02 less than 0.05 
#beta-0=0.7410
#beta-1=0.0403
pred3 = model3.predict(pd.DataFrame(wcat['Waist']))
pred3_at = np.exp(pred3)
#CHeck wcat and pred3_at from vvariable explorer 
#Regression Line
#scatter diagram 
plt.scatter(wcat.Waist, np.log(wcat.AT))
plt.plot(wcat.Waist,pred3,"r")
plt.legend(['observed data','predicted line_model3'])
plt.show()

#Error calculation 
res3 = wcat.AT-pred3_at
mse3 = np.mean(res3)
res_sqr3 = res3*res3
mse3 = np.mean(res_sqr3)
rmse3 = np.sqrt(mse3)
rmse3

#RMSE is 38.53 
#Polynomial transformation 
#x = Waist, x^2 = Waist*Waist, y=log(at) # Here we are adding polynomial 
#component in the model, due to adding polynomial component we cannot 
#calculate covariance 
model4 = smf.ols('np.log(AT)~Waist+I(Waist*Waist)', data=wcat).fit()
model4.summary()
#R square is 0.779<0.85 , p value is 0<0.005 

#p=0.000<0.05 hence acceptable 
#beta-0 = -7.8241 
#beta-1 = 0.2289 
pred4 = model4.predict(pd.DataFrame(wcat.Waist))
pred4
pred4_at = np.exp(pred4)
pred4_at
##############################################################################
#Regression Line 
plt.scatter(wcat.Waist,np.log(wcat.AT))
#Data is linearly scattered , direction: positive strength: poor 
plt.scatter(wcat.Waist,np.log(wcat.AT))
plt.plot(wcat.Waist,pred4,'r')
plt.legend(['Predicted line','Observed data_model3'])
plt.show()
###############################################################################
#Error calculation 
res4 = wcat.AT-pred4_at

res_sqr4 = res4*res4
mse4 = np.mean(res_sqr4)
rmse4 = np.sqrt(mse4)
rmse4
#32.24
#Among all the model model4 is the best 
####################################################
data = {"model":pd.Series(['SLR','"log_model','Exp_model','Poly_model'])}
data
table_rmse = pd.DataFrame(data)
table_rmse
##############################################################################
#We have to generalize the best model 
from sklearn.model_selection import train_test_split
train,test = train_test_split(wcat, test_size=0.2)
plt.scatter(train.Waist,np.log(train.AT))
plt.scatter(test.Waist,np.log(test.AT))
plt.show()
final_model = smf.ols('np.log(AT)~Waist+I(Waist*Waist)',data=wcat).fit()
#Y is log(AT) and X = Waist
final_model.summary()
#R-squared = 0.779<0.85, there is scope of improvemnet 
#p = 0.00<0.05 hence acceptable
#beta-0 = -7.8241 
#beta-1 = 0.22289 
test_pred = final_model.predict(pd.DataFrame(test))
test_pred_at = np.exp(test_pred )
test_pred_at
###########################################################################
train_pred = final_model.predict(pd.DataFrame(train))
train_pred_at = np.exp(train_pred)
train_pred_at
###########################################################################
#Evaluation on test data 
test_err = test.AT-test_pred_at
test_sqr = test_err*test_err
test_mse = np.mean(test_sqr)
test_rmse = np.sqrt(test_mse)
test_rmse
##########################################################
#Evaluation on train data
train_res = train.AT-train_pred_at
train_sqr = train_res*train_res
train_mse = np.mean(train_sqr)
train_rmse = np.sqrt(train_mse)
train_rmse

#########################################################
test_rmse>train_rmse
#hence the model is Overfit




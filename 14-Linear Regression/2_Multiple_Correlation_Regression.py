# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 14:59:35 2024

@author: sumit


Title:- Multiple Correlation Regression Analysis
"""
import pandas as pd 
import numpy as np 
import seaborn as sns
cars = pd.read_csv('cars.csv')
#EDA 
#1. Measure the central tendancy 
# 2. Measures of dispersion
# 3. Third Moment Buisness Analysis -> skewness
# 4. Fourth Moment Buisness decision -> kurtosis
#5. Probability distribution 
#6.Graphical representation(HIstogram,Boxplot)
cars.describe()
#Graphical representation 
import matplotlib.pyplot as plt
plt.bar( height=cars.HP, x=np.arange(1,82,1))
sns.distplot(cars.HP)
#data is right skwed
plt.boxplot(cars.HP)
#There are several outliers in HP columns 
#similar operations are expected for other  three columns 
sns.distplot(cars.MPG)
#dat is slightly left distributed or skwed 
plt.boxplot(cars.MPG)
#There are no  outliers 
sns.displot(cars.VOL)
#dat is slightly lefrt distributed 
sns.boxplot(cars.VOL)
sns.displot(cars.SP)
#data is slightly right distributed 
plt.boxplot(cars.SP)
#There are several outliers  
sns.distplot(cars.WT)
plt.boxplot(cars.WT)
#There are several outliers 
#Now let us plot joint plot, joint plot is to show scatter 
#histogram 
import seaborn as sns
sns.jointplot(x=cars['HP'],y=cars['MPG'])

#Now let us plot count plot 
plt.figure(1,figsize=(16,10))
sns.countplot(cars['HP'])
#count plot shows how many times the each  alue occured in the data 
#92 HP value occured 7 times
###############################################################################
#QQ Plot
from scipy import stats 
import pylab
stats.probplot(cars.MPG, dist="norm",plot=pylab)
plt.show()

#MPG data is normally distributed 
#There are 10 scatter plots need to be plotted, one by one 
# to plot, so we can use pair plot 
import seaborn as sns
sns.pairplot(cars.iloc[:,:])
#For MPG-HP 
# linearity: ,direction:negative ,strength:moderate 

#you can check the collinearity probelm between the input 
#you can check plot between SP and HP, then are strongly correlated 
#same way chekc WT and VOl, it is also strongly correlated 
#NOw let us check r value between variabeles
cars.corr()
#you can SP and HP , r value is 0.97 and same way  
#yoju can check WT and VOL, it has got 0.999 which is great 

#Now although we observed strongly correlated pairs,  still we will go for
#linear regresion 
import statsmodels.formula.api as smf
ml1 = smf.ols('MPG~WT+VOL+SP+HP',data=cars).fit()
ml1.summary()
#R sq. value observed is 0.771<0.85 
#p-values of WT and VOL is 0.814 and 0.556 which is very high 
#it means it is greater than 0.05. WT and VOL columns
#we need to ignore 
#or delete. Instead deleting 81 entries,
#Let us check row wise outliers 
#identifying is there any influential value 
#To check you can use influential index 
import statsmodels.api as sm 
sm.graphics.influence_plot(ml1)
#76 is the value which has got outliers 
#go to data frame and check 76th entry 
#Let us delte that entry
cars_new = cars.drop(cars.index[[76]])
#again applu regression to cars_new
ml_new = smf.ols('MPG~WT+VOL+HP+SP',data=cars_new).fit()
ml_new.summary()

#R-square value is 0.819 but p values are same,  hence not sco 
#now next option is delte the column byt 
#question is which column is to be deleted
#we have already checked correlation factor r 
#VOL has got -0.529 and fir WT = -0.526
#WT is less hence can be deleted 

#another approach is to check the collinearity 
#r square is giving that value 
#we will have to apply regression w.r.t x1 and input 
#as x2,x3 and x4 so on so forth 
rsq_hp = smf.ols('HP~WT+VOL+SP',data=cars).fit().rsquared
vif_hp = 1/(1-rsq_hp)
vif_hp

#
#
rsq_wt = smf.ols('WT~HP+VOL+SP',data=cars).fit().rsquared
vif_wt = 1/(1-rsq_wt)
vif_wt

rsq_vol = smf.ols('VOL~HP+VOL+SP',data=cars).fit().rsquared
vif_vol = 1/(1-rsq_vol)
vif_vol

rsq_sp = smf.ols('SP~HP+VOL+SP',data=cars).fit().rsquared
vif_sp = 1/(1-rsq_sp)
vif_sp

#vif_WT = 639.53 and vif_vol=638.80 hence vif_wt 
#is greater, thumb rule is vif should not be greater than 1 
#sorting the values in dataframe 
d1={'Variables':['HP','WT','VOL','SP'],'VIF':[vif_hp,vif_wt,vif_vol,vif_sp]}
vif_frame=pd.DataFrame(d1)
vif_frame

# let us drop WT and apply correlation to remailing three
final_ml = smf.ols('MPG~VOL+SP+HP',data=cars).fit()
final_ml.summary()
#R square is 0.770 and p values 0.00, 0.012 < 0.05 

final_ml2 = smf.ols('MPG~VOL+SP+HP',data=cars_new).fit()
final_ml2.summary()
#R square is 0.819 and p values 0.00, 0.819 < 0.05 

#prediction 
pred=final_ml.predict(cars_new)
#QQ Plot 
res = final_ml.resid 
sm.qqplot(res)
plt.show()

#This qq plot is on residual which is obtained on training 
#errors are obtained on test data 
stats.probplot(res,dist="norm",plot=pylab)
plt.show()

#Let us plot the residual plot, which takes the residuals 
#and the data 
sns.residplot(x=pred,y=cars_new.MPG, lowess=True)
plt.xlabel('FItted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

#splitting the daa into train and test data
from sklearn.model_selection import train_test_split
cars_train,cars_test = train_test_split(cars_new, test_size=0.2)
#preparing the model on train data 
model_train = smf.ols('MPG~VOL+SP+HP', data=cars_train).fit()
model_train.summary()
test_pred = model_train.predict(cars_test)
#test errors 
test_error = test_pred-cars_test.MPG
test_rmse = np.sqrt(np.mean(test_error*test_error))
test_rmse












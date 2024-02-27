# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 08:19:23 2024

@author: sumit
"""

import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
forest = pd.read_csv('forestfires.csv')

forest.dtypes
# month             object
# day               object
# FFMC             float64
# DMC              float64
# DC               float64
# ISI              float64
# temp             float64
# RH                 int64
# wind             float64
# rain             float64
# area             float64
# dayfri             int64
# daymon             int64
# daysat             int64
# daysun             int64
# daythu             int64
# daytue             int64
# daywed             int64
# monthapr           int64
# monthaug           int64
# monthdec           int64
# monthfeb           int64
# monthjan           int64
# monthjul           int64
# monthjun           int64
# monthmar           int64
# monthmay           int64
# monthnov           int64
# monthoct           int64
# monthsep           int64
# size_category     object

###############################################################################
#---------------EDA------------------------- 
forest.shape
#(571,31)
forest.dtypes

plt.figure(1,figsize=(16,10))
sns.countplot(data=forest,x=forest.month)
#AUg to sept has highest value 
sns.countplot(data=forest,x=forest.day)
#friday sunday and saturday have highest value
sns.distplot(forest.FFMC)
#data isnormal and slight left skewed

sns.boxplot(forest.FFMC)
#There are several outliers

sns.distplot(forest.DC)
# data is normal and slight left skewed
sns.boxplot(forest.DC)
#There are outliers

sns.distplot(forest.RH)
# data is normal and slight left shewed
sns.boxplot(forest.RH)
#there are outliers

sns.distplot(forest.wind)
#data is normal and slight right skewed 
sns.boxplot(forest.wind)
#There are outliers 
sns.distplot(forest.rain)
#data is normal 
sns.boxplot(forest.rain)
#There are outliers 

sns.displot(forest.area)
#data is normal 
sns.boxplot(forest.area)
#There are outliers 

#Now let us check the highest fire in sq.KM
forest.sort_values(by='area', ascending=False).head(5)

highest_fire_area = forest.sort_values(by='area', ascending=True)

plt.figure(figsize=(8,6))
plt.title("Temperature vs area of Fire ")
plt.bar(highest_fire_area['temp'],highest_fire_area['area'])
plt.xlabel("Temperature")
plt.ylabel("Area per km-sq")
plt.show()

#Once the fire starts, almost 1000+ sq area's 
#temperature goes beyond 25 and 
#around 750km area is facing temp 30+ 
#Now let us check the highest rain in the forest 
highest_rain=forest.sort_values(by='rain',ascending=False)[['month','day','rain']].head(5)
highest_rain 

#highest rain observed in the month of aug 
#Let us check highest and lowest temperature in month and 
highest_temp = forest.sort_values(by='temp', ascending=False)[['month','day','rain']].head(5)
highest_temp

lowest_temp = forest.sort_values(by='temp', ascending=True)[['month','day','rain']].head(5)
lowest_temp
#highest temperature is in aug 
#lowest temp is in dec 

forest.isna().sum()
#So here are no null values 

########################################################################
#sall.dtypes 

from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()
forest.month = labelencoder.fit_transform(forest.month)
forest.day = labelencoder.fit_transform(forest.day)
forest.size_category = labelencoder.fit_transform(forest.size_category)

forest.dtypes 
from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['month'])
df_t = winsor.fit_transform(forest[['month']])
sns.boxplot(df_t.month)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['FFMC'])
df_t = winsor.fit_transform(forest[['FFMC']])
sns.boxplot(df_t.FFMC)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['RH'])
df_t = winsor.fit_transform(forest[['RH']])
sns.boxplot(df_t.RH)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['wind'])
df_t = winsor.fit_transform(forest[['wind']])
sns.boxplot(df_t.wind)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['rain'])
df_t = winsor.fit_transform(forest[['rain']])
sns.boxplot(df_t.rain)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['area'])
df_t = winsor.fit_transform(forest[['area']])
sns.boxplot(df_t.area)

from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['DC'])
df_t = winsor.fit_transform(forest[['DC']])
sns.boxplot(df_t.DC)
#Copy paste for all the featureshaving outliers 

#####################################################################

tc = forest.corr()
tc
fig,ax = plt.subplots()
fig.set_size_inches(200,10)
sns.heatmap(tc,annot=True)
#all the variables are moderately correlated with size_category

from sklearn.svm import SVC 
from sklearn.model_selection import train_test_split

train, test = train_test_split(forest, test_size=0.3)
train_X = train.iloc[:,:30]
train_y = train.iloc[:,30]
test_X = test.iloc[:,:30]
test_y = test.iloc[:,30]

#kernel linear 
model_linear = SVC(kernel = "linear")
model_linear.fit(train_X,train_y)
pred_test_linear = model_linear.predict(test_X) 
np.mean(pred_test_linear == test_y)
#RBF 
model_rbf = SVC(kernel='rbf')
model_rbf.fit(train_X,train_y)
pred_test_rbf = model_rbf.predict(test_X)
np.mean(pred_test_rbf == test_y)

#Regularization
model_C = SVC(C=1)
model_C.fit(train_X,train_y)
model_C.score(test_X,test_y)

model_C = SVC(C=10)
model_C.fit(train_X,train_y)
model_C.score(test_X,test_y)

#Gamma
model_g = SVC(gamma=1)
model_g.fit(train_X,train_y)
model_g.score(test_X,test_y)

model_g = SVC(gamma=10)
model_g.fit(train_X,train_y)
model_g.score(test_X,test_y)

#Kernel
model_k = SVC(kernel='linear')
model_k.fit(train_X,train_y)
model_k.score(test_X,test_y)





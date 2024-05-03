# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 19:31:11 2024

@author: sumit
"""

import pandas as pd 
import numpy as np 
from sklearn import linear_model 
import matplotlib.pyplot as plt

df = pd.read_csv('calories_consumed.csv')

df.columns
# ['Weight gained (grams)', 'Calories Consumed']

df.rename({'Weight gained (grams)':'weight_gained_gm','Calories Consumed':'calories_consumed'},axis=1,inplace=True)
#The name of features has been converted in standard form.

#so there are no outliers in the data 

#lets visualize the data
plt.xlabel('weight_gained_gm')
plt.ylabel('calories_consumed')
plt.scatter(df.weight_gained_gm,df.calories_consumed,color='red',marker='*')
#From the graph we can see that there is a slight positive correlation

weight = df.drop({'weight_gained_gm'},axis=1)
calories = df.drop({'calories_consumed'},axis=1)

#Create linear regression object 
reg = linear_model.LinearRegression()
reg.fit(calories,weight)

reg.predict([[150]])

reg.predict([[250]])

reg.predict([[110]])



















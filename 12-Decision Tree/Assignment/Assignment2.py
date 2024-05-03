# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 19:41:26 2024

@author: sumit
"""

#1.Importing Libraraies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

#2.Load the Dataset
df=pd.read_csv("froud_check.csv")
df.columns
new_df=df.drop(['Urban'],axis='columns')
target=df['Urban']
new_df.head(5)
new_df.tail(5)
new_df.isnull().sum()
new_df.info()
new_df.describe()

#3.Visualization
#EDA
#scatterplot
plt.scatter(x=new_df['Marital.Status'],y=new_df['Taxable.Income'])
plt.xlabel("Position")
plt.ylabel("Experiance")
plt.title("Position vs Experiance")

#heatmap for coralation
sns.heatmap(new_df.corr(),annot=True)
plt.hist(df['Experiance'])
#Boxplot for Outliers
sns.boxplot(data=new_df)

#Model Building
from sklearn.model_selection import train_test_split
#train the data
X_train,X_test,y_train,y_test = train_test_split(new_df['Marital.Status'],new_df['Taxable.Income'],test_size=0.2)
len(X_train)

#choose the model
regressor=DecisionTreeRegressor()

#train the model
regressor.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

#testing the model
y_pred=regressor.predict(y_train.values.reshape(-1,1))
y_pred

#create a new dataframe and compare the values
comp=pd.DataFrame({"Actual Value":y_train,"Predicted Value":y_pred})
comp.head(5)
comp.tail(5)
sns.heatmap(comp.corr(),annot=True)

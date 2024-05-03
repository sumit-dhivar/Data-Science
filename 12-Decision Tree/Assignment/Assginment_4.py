# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 23:31:19 2024

@author: sumit

"""

'''
Business Objective
Minimize: To reduce costs, risks, or inefficiencies in a business process.

Maximize: To increase profits, efficiency, or positive outcomes.
"""
"""
Data Dictionary

 Features                                      Type             Relevance
0   Position of the employee                 Qualititative data  Relevant
1   no of Years of Experience of employee    Continious data     Relevant
2   monthly income of employee               Continious data     Relevant

"""

'''
import pandas as pd
import numpy as np
hr=pd.read_csv("HR_DT.csv")
hr.head(10)
hr.tail()
##########################
# 5 number summary
hr.describe()
##########################
#shape of the dataset
hr.shape
# 600 rows and 6 columns
##########################
hr.columns
'''Index(['Position of the employee', 'no of Years of Experience of employee',
       ' monthly income of employee'],
      dtype='object')'''
###########################
# check for null values
hr.isnull()
# False
############################
hr.isnull().sum()
# no null values
###########################
# Pair-Plot
import matplotlib.pyplot as plt
import seaborn as sns
plt.close();
sns.set_style("whitegrid");
sns.pairplot(hr);
plt.show()
# there are the some outlier are present in the dataset

###########################
hr.columns = [
    'Position_of_the_employee',
    'Years_Experience_employee',
    'monthly_income'
]

print(hr.columns)
# boxplot
# boxplot on Income column
sns.boxplot(hr.monthly_income)
# In Income column 1 outliers 

sns.boxplot(hr.Years_Experience_employee)
# In Population column no outliers

# boxplot on df column
sns.boxplot(hr)
# There is outliers on all columns

# histplot - show distributions of datasets
sns.histplot(hr['monthly_income'],kde=True)
# right skew and the distributed

sns.histplot(hr['Years_Experience_employee'],kde=True)
# right skew and the distributed

sns.histplot(hr,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# Data Preproccesing
hr.dtypes
# Some columns in int data types and some Object

# Identify the duplicates
duplicate=hr.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created
duplicate
# False
sum(duplicate)
#############
# Normalize data 
# Normalize the data using norm function
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
# Apply the norm_fun to data 
df1=norm_fun(hr.iloc[:,3:4])
df1
#########################
#combine the  all the feature in the df1
# in this we are changing the sequence of the dataset
hr['monthly_income']
df1['monthly_income']=hr['monthly_income']

hr['Years_Experience_employee']
df1["Years_Experience_employee"]=hr["Years_Experience_employee"]

hr[ 'Position_of_the_employee']
df1["Position_of_the_employee"]=hr[ 'Position_of_the_employee']

df1

####################################
# Converting into binary
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df1['Position_of_the_employee']= lb.fit_transform(df1['Position_of_the_employee'])

df1["Position_of_the_employee"].unique()
df1['Position_of_the_employee'].value_counts()
colnames=list(df1.columns)

predictors=colnames[:2]
target=colnames[2]



# Spliting data into training and testing data set
from sklearn.model_selection import train_test_split
train,test=train_test_split(df1,test_size=0.3)

from sklearn.tree import DecisionTreeClassifier as DT

model=DT(criterion='entropy')
model.fit(train[predictors], train[target])
preds_test=model.predict(test[predictors])
preds_test
pd.crosstab(test[target], preds_test,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_test==test[target])

# Now let us check accuracy on training dataset
preds_train=model.predict(train[predictors])
pd.crosstab(train[target], preds_train,rownames=['Actual'],colnames=['predictions'])
np.mean(preds_train==train[target])


# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:56:38 2024

@author: sumit

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("C:/Supervised_ML/Random_forest_Algo/Data_Sets/Fraud_check.csv")

#check the shape of the dataframe
df.shape
#(600, 6) --> contains 600 rows and 6 columns

#colums of the Dataframe
df.columns
"""Index(['Undergrad', 'Marital.Status', 'Taxable.Income', 'City.Population',
       'Work.Experience', 'Urban'],
      dtype='object') """

#what is the datatype of the columns
df.dtypes
"""
Undergrad          object
Marital.Status     object
Taxable.Income      int64
City.Population     int64
Work.Experience     int64
Urban              object
dtype: object
 """
 
#print starting row values
df.head

#Obtain the 5 number summary of the dataset
df.describe()

"""
        Taxable.Income  City.Population  Work.Experience
count      600.000000       600.000000       600.000000
mean     55208.375000    108747.368333        15.558333
std      26204.827597     49850.075134         8.842147
min      10003.000000     25779.000000         0.000000
25%      32871.500000     66966.750000         8.000000
50%      55074.500000    106493.500000        15.000000
75%      78611.750000    150114.250000        24.000000
max      99619.000000    199778.000000        30.000000
 """
 
# check for null values
df.isnull()
# False

df.isnull().sum()
# 0 no null values

# Now you can access the columns
print(df.columns)

# boxplot
# boxplot on Income column
sns.boxplot(df.Income)
# In Income column 1 outliers 

sns.boxplot(df.Population)
# In Population column no outliers

# boxplot on df column
sns.boxplot(df)
# There is outliers on all columns

# histplot - show distributions of datasets
sns.histplot(df['Income'],kde=True)
# right skew and the distributed

sns.histplot(df['Population'],kde=True)
# right skew and the distributed

sns.histplot(df,kde=True)
#The data is showing the skewness 
# most of the right skiwed data

# Data Preproccesing
df.dtypes
# Some columns in int data types and some Object

# Identify the duplicates
duplicate=df.duplicated()
# Output of this function is single columns
# if there is duplicate records output- True
# if there is no duplicate records output-False
# Series is created
duplicate
# False
sum(duplicate)
# sum is 0.

# Pair-Plot
plt.close();
sns.set_style("whitegrid");
sns.pairplot(df);
plt.show()

#performing descitization on the Taxable.Income Data based on given condition
#Taxable.Income <= 30000 ,into Risky or good
# Define the condition
condition = (df['Taxable.Income'] <= 30000)

# Create a new column 'Risk_Category' based on the condition
df['Label'] = np.where(condition, 'Risky', 'Good')

#now we convert into numeric data 
from sklearn.preprocessing import LabelEncoder
enc = LabelEncoder()
df['Label'] = enc.fit_transform(df['Label'])

#seprate the target variable
y = df.Label

X = df.drop(['Label'] , axis = "columns")

X.columns

X['Undergrad_n'] = enc.fit_transform(X['Undergrad'])
X['Marital.Status_n'] = enc.fit_transform(X['Marital.Status'])
X['Taxable.Income_n'] = enc.fit_transform(X['Taxable.Income'])
X['City.Population_n'] = enc.fit_transform(X['City.Population'])
X['Work.Experience_n'] = enc.fit_transform(X['Work.Experience'])
X['Urban_n'] = enc.fit_transform(X['Work.Experience'])

Xn = X.drop(['Undergrad', 'Marital.Status', 'Taxable.Income', 'City.Population', 'Work.Experience','Urban'] , axis = 'columns')
y


from sklearn.model_selection import train_test_split
Xn_train , Xn_test , y_train ,y_test = train_test_split(Xn,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=20)
#n_estimators : number of trees in the forest

model.fit(Xn_train , y_train)

model.score(Xn_test , y_test)
y_predicted = model.predict(Xn_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_predicted)
cm

#%matplotlib inline

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10,7))
sns.heatmap(cm , annot = True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


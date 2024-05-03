# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 15:01:16 2024

@author: sumit
"""

#Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#Load the Dataset
df=pd.read_csv("Diabetes.csv")
df.head(5)
df.info()
df.columns
df[' Class_variable']
df=df.rename({' Number of times pregnant':'pregnant',' Plasma glucose concentration':'plasma',
              ' Diastolic blood pressure':'BP',' Triceps skin fold thickness':'skin',
              ' 2-Hour serum insulin':'insulin',' Body mass index':'bmi',
              ' Diabetes pedigree function':'pedigree',' Age (years)':'age',' Class_variable':'output'},axis=1)

X=df.drop(['output'],axis=1)
y=X.a_output
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
X['a_output'] = lb.fit_transform(df['output'])
X.columns
#Data Dictionary
column_names = X.columns.tolist()
data_types = X.dtypes.tolist()
sample_values = X.iloc[0].tolist()  # Assuming the first row contains sample values

# Create a data dictionary
data_dictionary = pd.DataFrame({
    'Column Name': column_names,
    'Data Type': data_types,
    'Sample Value': sample_values
})
data_dictionary
#EDA
#scatterplot
plt.scatter(x=X['Price'],y=X['Income'])
plt.xlabel("Price")
plt.ylabel("Income")
plt.title("Price vs Income")

#heatmap for coralation
sns.heatmap(X.corr(),annot=True)

#Boxplot for Outliers
sns.boxplot(data=X)

#outlier Analysis
Q1 = X.quantile(0.25)
Q3 = X.quantile(0.75)

#Calculate the interquartile range (IQR)
IQR = Q3 - Q1
# Define a multiplier to determine the range within which data points are not considered outliers
multiplier = 1.5

# Define the lower and upper bounds for outliers detection
lower_bound = Q1 - multiplier * IQR
upper_bound = Q3 + multiplier * IQR

# Filter DataFrame to remove outliers
X[(X < lower_bound) | (X > upper_bound)] = pd.NA

X.isnull().sum()
X.shape
X.mean
X=X.fillna(value=X.mean)
from sklearn.model_selection import train_test_split
#train the data
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

model=RandomForestRegressor()
#training the model
model.fit(X_train,y_train)
#model.score(X_train,y_train)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix

# Assuming y_test and y_predicted are your actual and predicted target values respectively
cm = confusion_matrix(y_test, y_predicted)
print("Confusion Matrix:")
print(cm)


#Matplotlib Inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")




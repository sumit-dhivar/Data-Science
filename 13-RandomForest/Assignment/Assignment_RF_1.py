# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:14:01 2024

@author: sumit
"""
#Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#2.Load the Dataset
df=pd.read_csv("companyData.csv")
X=df.drop(['Sales','ShelveLoc','Urban','US'],axis='columns')
y=df['Sales']
X.head(5)
X.tail(5)
X.isnull().sum()
X.info()
X.describe()

#3.Visualization
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
plt.scatter(x=X['Population'],y=X['Price'])
plt.xlabel("Population")
plt.ylabel("Price")
plt.title("Price vs Income")

#heatmap for coralation
sns.heatmap(X.corr(),annot=True)

#Boxplot for Outliers
sns.boxplot(data=X)
##########################################################
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
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()

# Assuming y_test and y_predicted are your actual and predicted target values respectively
y_predicted=y_predicted.astype(int)
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




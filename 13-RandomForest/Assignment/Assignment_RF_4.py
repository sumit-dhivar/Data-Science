# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 16:38:32 2024

@author: sumit
"""

#Ensemble Techniques
#Random Forest
#confusion matrix when data is imbalace
#Importing the Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#Load the Dataset
df=pd.read_csv("HR_DT.csv")
df.head(5)
df.info()
df = df.rename(columns={'Position of the employee':'Position','no of Years of Experience of employee':'Experiance',' monthly income of employee': 'income'})
df['income']=df.income
from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
df['a_Postion'] = lb.fit_transform(df['Position'])

X=df.drop(['income','Position'],axis=1)
y=df.income            
X['a_Postion']=X['a_Postion'].astype(float)                                                                                                                                                      

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
plt.scatter(x=X['Position'],y=X['Experiance'])
plt.xlabel("Position")
plt.ylabel("Experiance")
plt.title("Position vs Income")
#heatmap for coralation
sns.heatmap(X.corr(),annot=True)

#Boxplot for Outliers
sns.boxplot(data=X)
from sklearn.model_selection import train_test_split
#tain the data
X_train,X_test,y_train,y_test = train_test_split(df.drop(['income'],axis=1),df.income,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
X_train=X_train.drop(['Position'],axis=1)
X_test=X_test.drop(['Position'],axis=1)

#n_estimators is the no of trees in the random forest
model.fit(X_train,y_train)

model.score(X_train,y_train)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm
'''
array([[10,  0,  0],
       [ 0, 11,  0],
       [ 0,  2,  7]], dtype=int64)

'''
#Matplotlib Inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
#######################################################################



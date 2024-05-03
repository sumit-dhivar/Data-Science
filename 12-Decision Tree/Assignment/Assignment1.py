# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 14:37:55 2024

@author: sumit
"""

"""
Buisness Objective:-
    Encourage a culture of data-driven decision-making within the
     organization. Use ongoing analysis to inform various business 
     functions, ensuring that strategies are continuously refined 
     based on real-time insights.

"""
#importing libraries 
import pandas as pd 


##---------EDA----------
df = pd.read_csv('Company_data.csv')
df.columns


#Arranging the columns of the data frame
df=df[['CompPrice', 'Income', 'Advertising', 'Population', 'Price','Age', 'Education','ShelveLoc','Urban', 'US','Sales']]

#Now we will split the input and target features seperately
X = df.iloc[:,0:10]
y=df['Sales']

#Now we will convert the categorical data into numerical data
from sklearn.preprocessing import  LabelEncoder 
X.columns
enc = LabelEncoder()
X['ShelvePos'] = enc.fit_transform(X['ShelveLoc'])
X['Urban_new'] = enc.fit_transform(X['Urban'])
X['US_new'] = enc.fit_transform(X['US'])
X['CompPrice_n'] = enc.fit_transform(X['CompPrice'])
X['Income_n'] = enc.fit_transform(X['Income'])
X['Advertising_n'] = enc.fit_transform(X['Advertising'])
X['Population_n'] = enc.fit_transform(X['Population'])
X['Price_n'] = enc.fit_transform(X['Price'])
X['Age_n'] = enc.fit_transform(X['Age'])
X = X.drop(['ShelveLoc','Urban','US','CompPrice', 'Income', 'Advertising', 'Population', 'Price', 'Age','Education'],axis=1)
y = round(y)

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#Let us removve the features which are continous

from sklearn import tree 
model = tree.DecisionTreeClassifier()

model.fit(X_train,y_train) 

pred = model.predict(X_test)

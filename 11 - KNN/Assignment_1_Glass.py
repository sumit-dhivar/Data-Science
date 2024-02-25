# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 21:44:01 2024

@author: sumit
"""

"""
1. Buisness Objective:- 
The primary business objective is to automate the classification of glass
 materials based on their elemental composition. This automation aims to
 enhance the efficiency and accuracy of the classification process,
 reducing the reliance on manual labor. The goal is to streamline the glass
 manufacturing plant's operations by leveraging data-driven approaches to 
 classify materials promptly and effectively. This automation should 
 contribute to quicker decision-making, enabling the company to meet 
 customer requirements more efficiently.
2. Data Dictionary:- 
'RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe' -> all of these are the 
atomic names of the element found in earth to be used to make glass
Type-> It is the type of glass made from different components 
"""
#Importing Libraries and Modules
import pandas as pd
import numpy as np 
glass = pd.read_csv('glass.csv') 

#-----------------------EDA--------------------------------
glass.info()

glass.isnull().sum()
#So it does not have any NULL value

#Now Let's Normalize the dataframe
def norm_func(i):
    return (i-i.min())/(i.max()-i.min())

glass_n = norm_func(glass.iloc[:,:9])

#So our new normalised dataframe is ready
#Let us now apply X as input and Y as Output  
X = np.array(glass_n.iloc[:,:])
y = np.array(glass['Type'])

#Now let us split the data into training and testing 
from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#let's try to select correct value of k 

from sklearn.neighbors import KNeighborsClassifier
acc=[]
for i in range(3,50,2):
    #Declare the model 
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train,y_train)
    train_acc = np.mean(neigh.predict(X_train) == y_train)
    test_acc = np.mean(neigh.predict(X_test) == y_test)
    acc.append([train_acc,test_acc])

import matplotlib.pyplot as plt
plt.plot(np.arange(3,50,2),[i[0]for i in acc],'ro-')
plt.plot(np.arange(3,50,2),[i[1]for i in acc],'bo-')


#let us check for k=3
knn=KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train,y_train)
pred=knn.predict(X_test)

from sklearn.metrics import accuracy_score
accuracy_score(pred, y_test)
pd.crosstab(pred, y_test)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result1 = classification_report(y_test, pred)
print (result1)
























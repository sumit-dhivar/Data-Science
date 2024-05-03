# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:30:39 2024

@author: sumit
"""
"""

1)Business Objective:
Enhance the organizational efficiency and well-being of the
 animals at the National Zoopark in India by implementing a
 comprehensive segregation strategy based on species-specific
 needs, behavioral patterns, and health considerations. This
 initiative aims to create optimal living conditions for each
 animal group, fostering their physical and mental health, while
 simultaneously improving the overall visitor experience through
 a more organized and educational exhibit layout.
2) Data Dictionary:
    ['animal name', 'hair', 'feathers', 'eggs', 'milk', 'airborne',
           'aquatic', 'predator', 'toothed', 'backbone', 'breathes', 'venomous',
           'fins', 'legs', 'tail', 'domestic', 'catsize'] -> All these
    are the features of an animal and are relevant for classification 
    animal name -> is not needed/relevant
    type is the animal type 
"""

import pandas as pd
import numpy as np
zoo = pd.read_csv('Zoo.csv')
zoo.columns

#As the animal name is not relevant we will drop it 
zoo = zoo.drop(['animal name'],axis=1)

X = np.array(zoo.iloc[:,:16])
y = np.array(zoo['type'])

from sklearn.model_selection import train_test_split 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

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






















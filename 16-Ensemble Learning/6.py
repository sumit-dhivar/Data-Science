# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:53:58 2024

@author: sumit
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
import warnings
warnings.filterwarnings('ignore')
from sklearn import datasets
iris = datasets.load_iris()

X,y= iris.data[: , 1:3],iris.target#taking entire data as training data

clf1 = LogisticRegression()
clf2 = RandomForestClassifier(random_state = 1)
clf3 = GaussianNB()
########################### 
print("After three classifier")
for clf , label in zip ([clf1 , clf2 , clf3] , ['Logistic Regression' , 'Random Forest model' , 'Naive Bayes model']):
    scores = model_selection.cross_val_score(clf, X , y , cv = 5 , scoring = 'accuracy')
    print("Accuracy : " , scores.mean() , "for " , label)

voting_clf_hard = VotingClassifier(estimators=[(label[0],clf1),(label[1],clf2),(label[2],clf3)],voting='hard')

voting_clf_soft = VotingClassifier(estimators=[(label[0],clf1),(label[1],clf2),(label[2],clf3)],voting='soft')

label_new = ['Logistic Regression' , 'Random Forest model' , 'Naive Bayes model','voting_clf_hard','voting_clf_soft']

for clf , label in zip ([clf1 , clf2 , clf3,voting_clf_hard,voting_clf_soft] ,label_new):
    scores = model_selection.cross_val_score(clf, X , y , cv = 5 , scoring = 'accuracy')
    print("Accuracy : " , scores.mean() , "for " , label)


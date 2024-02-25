# -*- coding: utf-8 -*-
"""
Created on Thu Feb  1 08:21:02 2024

@author: sumit
"""
print("Have a Good Day:)")

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 

df = pd.read_csv('salaries.csv') 
df.head()
inputs = df.drop('salary_more_then_100k',axis=1)
target = df['salary_more_then_100k'] 
from sklearn.preprocessing import  LabelEncoder 
le_company = LabelEncoder()
le_job = LabelEncoder()
le_degree = LabelEncoder()
inputs['company_n'] = le_company.fit_transform(inputs['company'])
inputs['job_n'] = le_job.fit_transform(inputs['job'])
inputs['degree_n'] = le_degree.fit_transform(inputs['degree'])
inputs_n = inputs.drop(['company','job','degree'],axis=1)
target 
from sklearn import tree 
model = tree.DecisionTreeClassifier()
model.fit(inputs_n,target) 
#Is slaary of google, computer Engineer Bachelors degree > 100k 
model.predict([[2,1,0]])
#Is salary of Google, Computer Engineer, Master Degree > 100k
model.predict([[2,1,1]])

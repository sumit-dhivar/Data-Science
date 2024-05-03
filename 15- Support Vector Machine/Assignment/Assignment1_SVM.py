# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:56:31 2024

@author: sumit

"""
import pandas as pd
import seaborn as sns
Train = pd.read_csv('SalaryData_Train.csv')
Test = pd.read_csv('SalaryData_Test.csv')

Train.columns

Train.shape

Train.info()

Train.isnull().sum()
#So here are no null values

Train.info()
 # 0   age            30161 non-null  int64 
 # 1   workclass      30161 non-null  object
 # 2   education      30161 non-null  object
 # 3   educationno    30161 non-null  int64 
 # 4   maritalstatus  30161 non-null  object
 # 5   occupation     30161 non-null  object
 # 6   relationship   30161 non-null  object
 # 7   race           30161 non-null  object
 # 8   sex            30161 non-null  object
 # 9   capitalgain    30161 non-null  int64 
 # 10  capitalloss    30161 non-null  int64 
 # 11  hoursperweek   30161 non-null  int64 
 # 12  native         30161 non-null  object
 # 13  Salary         30161 non-null  object
sns.boxplot(Train)
#So there are ouliers in all of the numeric feature
#Lets do winsorization on it
from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['age'])
df_t = winsor.fit_transform(Train[['age']])
sns.boxplot(df_t.age)

winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['educationno'])
df_t = winsor.fit_transform(Train[['educationno']])
sns.boxplot(df_t.educationno)

winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['hoursperweek'])
df_t = winsor.fit_transform(Train[['hoursperweek']])
sns.boxplot(df_t.hoursperweek)

from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()

Train.workclass = labelencoder.fit_transform(Train.workclass)
Train.education = labelencoder.fit_transform(Train.education)
Train.maritalstatus = labelencoder.fit_transform(Train.maritalstatus)
Train.occupation = labelencoder.fit_transform(Train.occupation)
Train.relationship = labelencoder.fit_transform(Train.relationship)
Train.race = labelencoder.fit_transform(Train.race)
Train.sex = labelencoder.fit_transform(Train.sex)
Train.native = labelencoder.fit_transform(Train.native)

#Now the same for the test dataset
Test.columns

Test.shape

Test.info()

Test.isnull().sum()
#So here are no null values

Test.info()
# 0   age            15060 non-null  int64 
# 1   workclass      15060 non-null  object
# 2   education      15060 non-null  object
# 3   educationno    15060 non-null  int64 
# 4   maritalstatus  15060 non-null  object
# 5   occupation     15060 non-null  object
# 6   relationship   15060 non-null  object
# 7   race           15060 non-null  object
# 8   sex            15060 non-null  object
# 9   capitalgain    15060 non-null  int64 
# 10  capitalloss    15060 non-null  int64 
# 11  hoursperweek   15060 non-null  int64 
# 12  native         15060 non-null  object
# 13  Salary         15060 non-null  object
sns.boxplot(Test)
#So there are ouliers in all of the numeric feature
#Lets do winsorization on it
from feature_engine.outliers import Winsorizer 
winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['age'])
df_t2 = winsor.fit_transform(Test[['age']])
sns.boxplot(df_t2.age)

winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['educationno'])
df_t2 = winsor.fit_transform(Test[['educationno']])
sns.boxplot(df_t2.educationno)

winsor = Winsorizer(capping_method = "iqr", tail='both',fold=1.5, variables=['hoursperweek'])
df_t2 = winsor.fit_transform(Test[['hoursperweek']])
sns.boxplot(df_t2.hoursperweek)

from sklearn.preprocessing import LabelEncoder 
labelencoder = LabelEncoder()

Test.workclass = labelencoder.fit_transform(Test.workclass)
Test.education = labelencoder.fit_transform(Test.education)
Test.maritalstatus = labelencoder.fit_transform(Test.maritalstatus)
Test.occupation = labelencoder.fit_transform(Test.occupation)
Test.relationship = labelencoder.fit_transform(Test.relationship)
Test.race = labelencoder.fit_transform(Test.race)
Test.sex = labelencoder.fit_transform(Test.sex)
Test.native = labelencoder.fit_transform(Test.native)

Train_X = Train.iloc[:,:13]
Train_y = Train.iloc[:,13]

Test_X = Test.iloc[:,:13]
Test_y = Test.iloc[:,13]

from sklearn.svm import SVC 
import numpy as np
#kernel linear 
model_linear = SVC(kernel = "linear")
model_linear.fit(Train_X,Train_y)
pred_test_linear = model_linear.predict(Test_X) 
np.mean(pred_test_linear == Test_y)





























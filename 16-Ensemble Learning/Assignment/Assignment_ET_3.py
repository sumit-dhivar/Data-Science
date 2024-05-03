# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 20:28:28 2024

@author: sumit
"""

#Importing Libraries
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,classification_report

#Load Dataset
data=pd.read_excel("Coca_Rating_Ensemble.xlsx")
data.head(5)
data.info()
data.columns
# Finding null values
data.isna().sum()
y = data['Rating']
from sklearn.preprocessing import LabelEncoder
#Conconverting into Binary
lb=LabelEncoder()
data['Company']=lb.fit_transform(data['Company'])
data['Name']=lb.fit_transform(data['Name'])
data['REF']=lb.fit_transform(data['REF'])
data['Cocoa_Percent']=lb.fit_transform(data['Cocoa_Percent'])
data['Company_Location']=lb.fit_transform(data['Company_Location'])
data['Bean_Type']=lb.fit_transform(data['Bean_Type'])
data['Origin']=lb.fit_transform(data['Origin'])

data.hist(bins=30,figsize=(20, 15), color='#005b96')

# Standardization:
data.columns

sns.boxplot(x=data['Cocoa_Percent'])  
#Many Outliers
sns.boxplot(x=data['Bean_Type'])  
sns.boxplot(x=data['Rating'])  

# Write code for Winsorizer
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['Cocoa_Percent','Rating','Bean_Type'])

df_t = winsor.fit_transform(data[['Cocoa_Percent','Rating','Bean_Type']])

sns.boxplot(data['Cocoa_Percent'])
sns.boxplot(df_t['Cocoa_Percent'])
sns.boxplot(data['Rating'])
sns.boxplot(df_t['Rating'])

# Now No Outliers are there
# Check skewness
skew_df = pd.DataFrame(data.select_dtypes(np.number).columns, columns=['Feature'])
skew_df['Skew'] = skew_df['Feature'].apply(lambda feature: skew(data[feature]))
skew_df['Absolute Skew'] = skew_df['Skew'].apply(abs)

skew_df['Skewed'] = skew_df['Absolute Skew'].apply(lambda x: True if x >= 0.5 else False)
skew_df

#fetal Charges colume is clearly skewed as we also saw in the histogram
for columns in skew_df.query("Skewed==True")['Feature'].values:
    data[columns]=np.log1p(data[columns])
    
data.head()
data1=data.copy()
data1=pd.get_dummies(data1)    
data1.head()

data2=data1.copy()
sc=StandardScaler()
data2[data1.select_dtypes(np.number).columns]=sc.fit_transform(data2[data1.select_dtypes(np.number).columns])
data2.drop(['Rating'],axis=1,inplace=True)
data2.head()


#Spliiting
data_f=data2.copy()
target=data['Rating']
X_train,X_test,y_train,y_test = train_test_split(data_f,target,stratify=target,random_state=42,test_size=0.2)
X_train=X_train.astype(float)
y_train=y_train.astype(float)
X_test=y_train.astype(float)
y_test=y_train.astype(float)
X_train = y_train.reshape(-1, 1)
y_train = y_train.reshape(-1, 1)
X_test = y_train.reshape(-1, 1)
y_test = y_train.reshape(-1, 1)

#Modeling
from sklearn.ensemble import AdaBoostClassifier
ada_clf=AdaBoostClassifier(learning_rate=0.02,n_estimators=5000)
ada_clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix
#Evaluation on Testing Data
confusion_matrix(y_test,ada_clf.predict(X_test))
accuracy_score(y_test,ada_clf.predict(X_test))

#Accuracy : 1.0
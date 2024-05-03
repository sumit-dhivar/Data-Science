# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 19:07:32 2024

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

#Load the Dataset
data=pd.read_excel("Ensemble_Password_Strength.xlsx")
data.isnull().sum()
data.dropna()
data.info()
data=data.drop(['characters_strength'],axis=1)

#Perform One Hot Encoding
from sklearn.preprocessing import LabelEncoder
#Conconverting into Binary
lb=LabelEncoder()
data['characters'] = data['characters'].astype(str)
X=data['characters']=lb.fit_transform(data['characters'])
y=data['characters_strength']

#Histogram
data.hist(bins=30,figsize=(20, 15), color='#005b96')

plt.boxplot(X)
plt.boxplot(y)
########################################################

#Heatmap
sns.heatmap(data.corr())
#############################################################

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
data2.drop(['characters_strength'],axis=1,inplace=True)
data2.head()


#Spliiting
data_f=data2.copy()
target=data['characters_strength']
target=target.astypes(int)

X_train,X_test,y_train,y_test = train_test_split(data_f,y,stratify=target,random_state=42,test_size=0.2)

#Modeling
from sklearn.ensemble import AdaBoostClassifier
ada_clf=AdaBoostClassifier(learning_rate=0.02,n_estimators=5000)
ada_clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix
#Evaluation on Testing Data
confusion_matrix(y_test,ada_clf.predict(X_test))
accuracy_score(y_test,ada_clf.predict(X_test))

#Accuracy:  0.855







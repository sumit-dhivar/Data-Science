# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 19:06:52 2024

@author: sumit

"""

#Importing Libaries
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
data=pd.read_csv("Tumor_Ensemble.csv")
data.info()
data.columns
# Finding null values
data.isna().sum()

target = data['dimension_worst']

sns.countplot(x=target,palette='winter')
plt.title('dimension_worst')
sns.heatmap(data.corr(), cmap='YlGnBu', annot=True, fmt='.2f')
data.columns
# Observations:
sns.countplot(x='id', data=data,hue='dimension_worst', palette='pastel')
plt.title(" ")

sns.countplot(x='diagnosis', data=data, hue='dimension_worst', palette='winter')
plt.title(" ")

sns.countplot(x=' Diastolic blood pressure', data=data, hue=' Class variable', palette='pastel')
plt.title(" ")

sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(2, figsize=(20, 13))
plt.suptitle("Distribution of twitter hashtags and collections")

ax1 = sns.histplot(x=' Diastolic blood pressure', data=data, hue=' Class variable', kde=True, ax=ax[0], palette='winter')
ax1.set(xlabel=' Diastolic blood pressure', title=' Distribution of Diastolic blood pressure')

ax2 = sns.histplot(x=' Plasma glucose concentration', data=data, hue=' Class variable', kde=True, ax=ax[0], palette='viridis')
ax2.set(xlabel=' Plasma glucose concentration', title='Distribution of fire based on target variable')

plt.show()

data.hist(bins=30,figsize=(20, 15), color='#005b96')

# Standardization:
data.columns

sns.boxplot(x=data['radius_mean'])   
# has 7 outliers  
sns.boxplot(x=data['perimeter_mean'])  
sns.boxplot(x=data['area_mean'])  
sns.boxplot(x=data['smoothness_se'])  
sns.boxplot(x=data['dimension_se'])  
sns.boxplot(x=data['radius_worst'])  
sns.boxplot(x=data['texture_worst'])  
sns.boxplot(x=data['perimeter_worst'])  
sns.boxplot(x=data['area_worst'])  
sns.boxplot(x=data['smoothness_worst'])  
sns.boxplot(x=data['compactness_worst'])  
sns.boxplot(x=data['concavity_worst'])  
sns.boxplot(x=data['symmetry_worst'])  


# Write code for Winsorizer
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=['radius_mean', 'texture_mean', 'perimeter_mean',
                           'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
                           'points_mean', 'symmetry_mean', 'dimension_mean', 'radius_se',
                           'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
                           'compactness_se', 'concavity_se', 'points_se', 'symmetry_se',
                           'dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
                           'area_worst', 'smoothness_worst', 'compactness_worst',
                           'concavity_worst', 'points_worst', 'symmetry_worst'])

df_t = winsor.fit_transform(data[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'points_mean', 'symmetry_mean', 'dimension_mean', 'radius_se',
       'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'points_se', 'symmetry_se',
       'dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst',
       'area_worst', 'smoothness_worst', 'compactness_worst',
       'concavity_worst', 'points_worst', 'symmetry_worst']])

sns.boxplot(data['radius_mean'])
sns.boxplot(df_t['radius_mean'])

sns.boxplot(data['texture_mean'])
sns.boxplot(df_t['texture_mean'])

sns.boxplot(data['perimeter_mean'])
sns.boxplot(df_t['perimeter_mean'])

sns.boxplot(data['area_mean'])
sns.boxplot(df_t['area_mean'])

# Now No Outliers are there

# Check skewness
skew_df = pd.DataFrame(data.select_dtypes(np.number).columns, columns=['Feature'])
skew_df['Skew'] = skew_df['Feature'].apply(lambda feature: skew(data[feature]))
skew_df['Absolute Skew'] = skew_df['Skew'].apply(abs)

skew_df['Skewed'] = skew_df['Absolute Skew'].apply(lambda x: True if x >= 0.5 else False)
skew_df

# fetal Charges column is clearly skewed as we also saw in the histogram
for column in skew_df.query("Skewed==True")['Feature'].values:
    data[column] = np.log1p(data[column])

data1 = data.copy()

data1 = pd.get_dummies(data1)

data1.head()

data2 = data1.copy()
sc = StandardScaler()
data2[data1.select_dtypes(np.number).columns] = sc.fit_transform(data2[data1.select_dtypes(np.number).columns])
data2.drop(['dimension_worst'], axis=1, inplace=True)
data2.head()
data.dtypes
# Split the Data into Features (X) and Target Variable (y)
X = data.drop(columns=['dimension_worst'])
X=X.drop({'diagnosis'},axis=1)
y = data['dimension_worst']

# Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.ensemble import AdaBoostRegressor
from sklearn.metrics import mean_squared_error
# Modeling
ada_reg = AdaBoostRegressor(n_estimators=500, learning_rate=0.1)
ada_reg.fit(X_train, y_train)

# Evaluation on Testing Data
y_pred = ada_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)
#Mean Squared Error: 3.039528457252829e-05

# Evaluation on Testing Data
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
y_train.max()
y_test.max()
# Convert continuous values into binary classes
y_train_binary = (y_train >= 0.18855210654459068).astype(int)
y_test_binary = (y_test >= 0.12000279239469637).astype(int)

# Initialize and train AdaBoostClassifier
ada_clf = AdaBoostClassifier(learning_rate=0.02, n_estimators=5000)
ada_clf.fit(X_train, y_train_binary)

# Predict on the test set
y_pred_binary = ada_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test_binary, y_pred_binary)
print("Accuracy:", accuracy)
#Accuracy: 0.9912280701754386
'''
An Accuracy of 0.9912 indicates that this AdaBoostClassifier model is performing very well on the test data
'''
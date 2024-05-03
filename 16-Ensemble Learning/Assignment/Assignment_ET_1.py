# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 16:42:39 2024

@author: sumit
"""
# Importing Libraries
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.model_selection import train_test_split 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Load Dataset
data = pd.read_csv("Diabeted_Ensemble.csv")
data.head()
data.columns
data.info()
# Finding null values
data.isna().sum()

target = data[' Class variable']

sns.countplot(x=target, palette='winter')
plt.title('Class Variable')
sns.heatmap(data.corr(), cmap='YlGnBu', annot=True, fmt='.2f')

# Observations:
sns.countplot(x=' Number of times pregnant', data=data, hue=' Class variable', palette='pastel')
plt.title("0 chnace based Ticket class")

sns.countplot(x=' Plasma glucose concentration', data=data, hue=' Class variable', palette='winter')
plt.title("0 chnace based Ticket class")

sns.countplot(x=' Diastolic blood pressure', data=data, hue=' Class variable', palette='pastel')
plt.title("0 chnace based Ticket class")

sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(2, figsize=(20, 13))
plt.suptitle("Distribution of twitter hashtags and collections")

ax1 = sns.histplot(x=' Diastolic blood pressure', data=data, hue=' Class variable', kde=True, ax=ax[0], palette='winter')
ax1.set(xlabel=' Diastolic blood pressure', title=' Distribution of Diastolic blood pressure')

ax2 = sns.histplot(x=' Plasma glucose concentration', data=data, hue=' Class variable', kde=True, ax=ax[0], palette='viridis')
ax2.set(xlabel=' Plasma glucose concentration', title='Distribution of fire based on target variable')

plt.show()

data.hist(bins=30, figsize=(20, 15), color='#005b96')

# Standardization:
data.columns

sns.boxplot(x=data[' Diastolic blood pressure'])   
# has 7 outliers  
sns.boxplot(x=data[' 2-Hour serum insulin'])  
sns.boxplot(x=data[' Body mass index'])  
# Many outliers
sns.boxplot(x=data[' Diabetes pedigree function'])  
# Many Outliers
sns.boxplot(x=data[' Age (years)'])  
# Many Outliers


# Write code for Winsorizer
from feature_engine.outliers import Winsorizer

winsor = Winsorizer(capping_method='iqr',
                    tail='both',
                    fold=1.5,
                    variables=[' Diastolic blood pressure', ' Triceps skin fold thickness',
                               ' 2-Hour serum insulin', ' Body mass index',
                               ' Diabetes pedigree function', ' Age (years)'])

df_t = winsor.fit_transform(data[[' Diastolic blood pressure', ' Triceps skin fold thickness',
                                  ' 2-Hour serum insulin', ' Body mass index',
                                  ' Diabetes pedigree function', ' Age (years)']])

sns.boxplot(data[' Diastolic blood pressure'])
sns.boxplot(df_t[' Diastolic blood pressure'])

sns.boxplot(data[' 2-Hour serum insulin'])
sns.boxplot(df_t[' 2-Hour serum insulin'])

sns.boxplot(data[' Body mass index'])
sns.boxplot(df_t[' Body mass index'])

sns.boxplot(data[' Diabetes pedigree function'])
sns.boxplot(df_t[' Diabetes pedigree function'])

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
data2.drop([' Class variable_YES'], axis=1, inplace=True)
data2.head()

# Splitting
data_f = data2.copy()
target = data[' Class variable']
target = target.map({'YES': 1, 'NO': 0})

X_train, X_test, y_train, y_test = train_test_split(data_f, target, stratify=target, random_state=42, test_size=0.2)

# Modeling
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate=0.02, n_estimators=5000)
ada_clf.fit(X_train, y_train)

# Evaluation on Testing Data
from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(y_test, ada_clf.predict(X_test))
accuracy_score(y_test, ada_clf.predict(X_test))

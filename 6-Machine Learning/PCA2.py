# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 15:41:27 2023

@author: sumit
"""

"""
Data Dictionary:- 
Data Dictionary:- 
Feature                     Description                                              Type                   Relevance
age ;-             Age of the individual                                        Quantitative                relevant
sex:-           Gender of the individual (0 = female, 1 = male)                 Quantitative,Nominal        relevant
cp :-            	Chest pain type (0-3)                                       Quantitative,Nominal
trestbps :-              Resting blood pressure (mm Hg)                         Quantitative
chol:-        	Serum cholesterol (mg/dl)                                       Quantitative
fbs :-        Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)             Quantitative,Nominal
restecg:-           Resting electrocardiographic results (0-2)                  Quantitative,Nominal
thalach:-        Maximum heart rate achieved                                    Quantitative
exang:-     Exercise-induced angina (1 = yes; 0 = no)                           Quantitative,Nominal
oldpeak :-  ST depression induced by exercise relative to rest                  Quantitative
slope:-             Slope of the peak exercise ST segment                       Quantitative,Nominal
ca:-               Number of major vessels colored by fluoroscopy (0-3)         Quantitative
thal:-          Thalassemia type                                                Quantitative,Nominal

Buisness Objective:- To develop a medicine effective for all types of patients
Buisness Constraint:- 

"""
###############################################################################
#Importing Libraries, Files
import pandas as pd
df = pd.read_csv('heart.csv')

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD 
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 
###############################################################################

df.dtypes
"""
age           int64
sex           int64
cp            int64
trestbps      int64
chol          int64
fbs           int64
restecg       int64
thalach       int64
exang         int64
oldpeak     float64
slope         int64
ca            int64
thal          int64
"""
df.columns
"""
Index(['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
       'exang', 'oldpeak', 'slope', 'ca', 'thal'],
      dtype='object')

13 features
""" 
df.info()
# 0   age       1025 non-null   int64  
# 1   sex       1025 non-null   int64  
# 2   cp        1025 non-null   int64  
# 3   trestbps  1025 non-null   int64  
# 4   chol      1025 non-null   int64  
# 5   fbs       1025 non-null   int64  
# 6   restecg   1025 non-null   int64  
# 7   thalach   1025 non-null   int64  
# 8   exang     1025 non-null   int64  
# 9   oldpeak   1025 non-null   float64
# 10  slope     1025 non-null   int64  
# 11  ca        1025 non-null   int64  
# 12  thal      1025 non-null   int64  

"""Here we can see that only oldpeak is in float64 while others are in int64, we won't change the data type here"""

df.shape
#(1025, 13)

#Now let's Check for the null values
df.isnull().sum()
"""So we are having no null value in any of the features"""
df_desc = df.describe()
"""
-> the average age is 54.4341
-> the average Resting blood pressure is 131.612
-> the avg serum cholestrol is 51.5925
-> the average heart rate is 23.0057
-> the average ST depression induced by exercise relative to rest is 1.175
"""
df['sex'].value_counts()
#Here we can see that the mens have more rate of having a heart attack than woman it is around 69.5% almost 70%.

df['age'].value_counts().head(10)
"""
age  count
58    68
57    57
54    53
59    46
52    43
51    39
56    39
62    37
60    37
44    36

Here you can see the no of patients from top 10 age groups i.e
the patients of age 58 are 68 and so on.
"""
df['age'].value_counts().tail(10)
"""
age  count
38    12
71    11
40    11
69     9
37     6
34     6
29     4
76     3
77     3
74     3

Here you can see the no of patients from last 10 age groups i.e
the patients of age 77 are 3 and so on.
"""

df['cp'].value_counts()
#So the most of the patients does not feel chest pain.
plt.rcParams['figure.figsize'] = (12,6)
sns.boxplot(df)
#Here we can see that we are having outliers in some features and we need to treat them 
#but before that we will find the correlation between the features so that we can reduce the 
#number of features by removing the uneccesary features
#So for these we will plot the heatmap
sns.heatmap(df.corr(),cmap='coolwarm', annot=True, fmt='.2f')
#We eill select the featurs which are having the positive corelation among each othr for furrther analysis i.e >0.2
#age - ca,oldpeak, chol, trestbps, 
#cp - thalach
#thalach - slope
#exang - oldpeak
#oldpeak - ca 

#i.i we will select age,ca,oldpeak,chol,trestbps,cp,thalach,exang
df_new = df[['age','cp','trestbps','chol','thalach','exang','oldpeak','ca']]
df_new

sns.boxplot(df_new)
#######################################################################
#Now let's perform univariate analysis
#age
df_new['age'].describe()
#From this it is clear that it is not having any oulier 
#the avg age is 54
#the youngest patient is 29 yr old
#the oldest one is 77 yr old 
sns.boxplot(df_new['age'])

sns.histplot(df_new['age'],kde=True)
#This is a normal curve with no outlier 
#=======================================================================
#cp
df_new['cp'].describe()

sns.boxplot(df_new['cp'])

sns.histplot(df_new['cp'],kde=True)
#=========================================================================
#trestbps
sns.boxplot(df_new['trestbps'])
#Here we can see some outliers letss remove those outliers by treating them with winsorization technique
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['trestbps']
                  )

df_new['trestbps'] = winsor.fit_transform(df_new[['trestbps']])
sns.boxplot(df_new['trestbps'])
#Now we cann see thaat the outliers havbeen removed

sns.histplot(df_new['trestbps'],kde=True)
#Also we can see that it has become a normal curve.
#========================================================================
#chol
sns.boxplot(df_new['chol'])
#Here we can see some outliers letss remove those outliers by treating them with winsorization technique
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['chol']
                  )
df_new['chol'] = winsor.fit_transform(df_new[['chol']])

sns.boxplot(df_new['chol'])
#Now we cann see thaat the outliers havbeen removed
sns.histplot(df_new['chol'],kde=True)
#Also we can see that it has become a normal curve.
#========================================================================
#thalach 
sns.boxplot(df_new['thalach'])
#Here we can see some outliers letss remove those outliers by treating them with winsorization technique
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['thalach']
                  )
df_new['thalach'] = winsor.fit_transform(df_new[['thalach']])
sns.boxplot(df_new['thalach'])
#Now we cann see thaat the outliers havbeen removed
sns.histplot(df_new['thalach'],kde=True)
#Also we can see that it has become a normal curve also it is slightly left skwed.
 
#========================================================================
#oldpeak 
sns.boxplot(df_new['oldpeak'])
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['oldpeak']
                  )
df_new['oldpeak'] = winsor.fit_transform(df_new[['oldpeak']])
sns.boxplot(df_new['oldpeak'])
#Now we cann see thaat the outliers havbeen removed
sns.histplot(df_new['oldpeak'],kde=True)

#=========================================================================
#ca 
sns.boxplot(df_new['ca'])
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['ca']
                  )
df_new['ca'] = winsor.fit_transform(df_new[['ca']])
sns.boxplot(df_new['ca'])
#Now we cann see thaat the outliers havbeen removed
sns.histplot(df_new['ca'],kde=True)

##############################################################################
#Hierarchical Clustering
z=linkage(df_new, method='complete',metric='euclidean') 
plt.figure(figsize=(15,8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')
sch.dendrogram(z,leaf_rotation=90,leaf_font_size=10)
plt.show()

from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_new)
#apply labels to the clusters 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_) 
df_new['Clust'] =  cluster_labels
df_new.columns
df_final = df_new.loc[:,['Clust','age', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak', 'ca']]
df_final.iloc[:,1:].groupby(df_new.Clust).mean()
df_final.to_csv('D:/ANACONDA/Data Science/Assignment/Heirarchical2.csv')
#==============================================================================
#K-Means Clustering 
TWSS = []
k = list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_new)
    
    TWSS.append(kmeans.inertia_)#total within sum of square 
   
TWSS
#As k value increases the TWSS value decreases 
plt.plot(k,TWSS,'ro-');
plt.xlabel('No_of_clusters');
plt.ylabel('Total_within_SS');
#From the Scree Plot we can decide that 3 cluster is the best possible .
model = KMeans(n_clusters=3)    
model.fit(df_final)    
model.labels_ #This shows that the data point is in which cluster   
mb = pd.Series(model.labels_)    
df_final['Clust'] = mb    
df_final.head()  
df_final.columns  
df_final = df_final.loc[:,['Clust', 'age', 'cp', 'trestbps', 'chol', 'thalach', 'exang', 'oldpeak',
       'ca']]    
x = df_final.iloc[:,1:].groupby(df_final.Clust).mean()  
df_final.to_csv('D:/ANACONDA/Data Science/Assignment/kmeans2.csv',encoding='utf-8')

#===============================================================================
#Performing PCA 
df_ = df_new.drop({'Clust'},axis=1)
U,d,Vt = svd(df_)
svd = TruncatedSVD(n_components=3)
svd.fit(df_)
result = pd.DataFrame(svd.transform(df_))
result.head()
result.columns="pc0","pc1","pc2"
result.head()

#Scatter Diagram 

plt.scatter(x=result.pc0, y=result.pc1)














# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 10:30:23 2023

@author: sumit
"""

"""
Data Dictionary:- 
Feature                     Description                                              Type           Relevance
Type ;-             The class of the wine (target variable)                     Quantitative        relevant
Alcohol:-           The alcohol content in the wine (% by volume)               Quantitative        relevant
Malic :-            Malic acid content in the wine (g/l)                        Quantitative
Ash :-              Ash content in the wine (g/l)                               Quantitative
Alcalinity:-        Alcalinity of ash in the wine (measured in mEq/l)           Quantitative
Magnesium :-        Magnesium content in the wine (mg/dm^3)                     Quantitative
Phenols:-           Total phenols content in the wine (mg/g)                    Quantitative
Flavanoids:-        Flavanoids content in the wine (mg/g)                       Quantitative
Nonflavanoids:-     Non-flavanoid phenols content in the wine (mg/g)            Quantitative
Proanthocyanins :-  Proanthocyanins content in the wine (mg/g)                  Quantitative
Color:-             Intensity of color in the wine                              Quantitative
Hue:-               Hue of the wine color (measured in degrees)                 Quantitative
Dilution:-          OD280/OD315 of diluted wines, a measure of color intensity  Quantitative
Proline:-           Proline content in the wine (mg/dm^3)                       Quantitative

Constraint :- 
Objective :- To imporve the quality of the wine
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from scipy.linalg import svd
from sklearn.decomposition import TruncatedSVD 

df = pd.read_csv('wine.csv')
df
df.columns
# 'Type', 'Alcohol', 'Malic', 'Ash', 'Alcalinity', 'Magnesium', 'Phenols',
#        'Flavanoids', 'Nonflavanoids', 'Proanthocyanins', 'Color', 'Hue',
#        'Dilution', 'Proline'
df.shape
#(178, 14) 

df.info()
#  #   Column           Non-Null Count  Dtype  
# ---  ------           --------------  -----  
#  0   Type             178 non-null    int64  
#  1   Alcohol          178 non-null    float64
#  2   Malic            178 non-null    float64
#  3   Ash              178 non-null    float64
#  4   Alcalinity       178 non-null    float64
#  5   Magnesium        178 non-null    int64  
#  6   Phenols          178 non-null    float64
#  7   Flavanoids       178 non-null    float64
#  8   Nonflavanoids    178 non-null    float64
#  9   Proanthocyanins  178 non-null    float64
#  10  Color            178 non-null    float64
#  11  Hue              178 non-null    float64
#  12  Dilution         178 non-null    float64
#  13  Proline          178 non-null    int64 

"""As we can see that the Magnesium and the Proline is in int64 dtype lets change it to float64 dtype """

df[['Magnesium','Proline']]=df[['Magnesium','Proline']].astype(float)

#Now let's Check for the null values
df.isnull().sum()
"""So we are having no null value in any of the features"""

df_desc = df.describe()

plt.rcParams['figure.figsize'] = (12,6)
sns.boxplot(df)

#Now let's find the correlation between the features and remove the unnneccessary featuress
sns.heatmap(df.corr(),cmap='coolwarm', annot=True, fmt='.2f')
#So Now we will take the features which are having correlation greater than 0.5
#Type - Alcalinity
#Alcohol - Color
#Alcohol - Proline
#Phenols - Proline
#Phenols - Dilution 
#Phenols - Proanthocyanins 
#Phenols - flavanoids 
#Flavanoids - Dilution
#Flavanoids - Hue
#Flavanoids - Proanthocyanins 
#Proanthocyanins - Dilution 
# Hue  - Dilution 

"""So from the above relation which are having positive correlation between them, 
we will use Alcohol, color, proline, phenols, dilution, Proanthocyanins, Flavanoids, Hue
i.e 8 features"""

df_new = df[['Alcohol','Color','Proline','Phenols','Dilution','Proanthocyanins','Flavanoids','Hue']]

sns.boxplot(df_new)

#Before performing univariate analysis lets standardize our data 
#So it will be convenient for us to compare the data.

scalar=StandardScaler()
df_1 = scalar.fit_transform(df_new) 
df_final = pd.DataFrame(df_1,columns=['Alcohol','Color','Proline','Phenols','Dilution','Proanthocyanins','Flavanoids','Hue'])
#######################################################################
#Now let's perform univariate analysis
#==================================================================
#Alcohol
df_final['Alcohol'].describe()
"""
->As we can see that he std deviation is low we can infer that the alcohol content in each type of wine is now varing a lot
mean    -8.382808e-16
std      1.002821e+00
min     -2.434235e+00
max      2.259772e+00
"""
sns.boxplot(df_final['Alcohol'])
#So we are not having any outlier in this feature

sns.histplot(df_new['Alcohol'],kde=True)
#This is almost a normal curve and can be used for clustering directly

#==========================================================================
#Color 
df_final['Color'].describe()
"""
mean     2.494883e-17
std      1.002821e+00
min     -1.634288e+00
max      3.435432e+00
"""
sns.boxplot(df_final['Color'])
#So here we are having 3 outliers in the data, let's treat them through winsorization
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Color']
                  )
df_final['Color'] = winsor.fit_transform(df_final[['Color']])

sns.boxplot(df_final['Color'])
#Now the ouliers are removed lets check the distribution 

sns.histplot(df_final['Color'],kde=True)
#The featture is now ready for the clustering , it is normal with slight right skew
#=================================================================================
#Proline
df_final['Proline'].describe()
"""
mean    -1.596725e-16
std      1.002821e+00
min     -1.493188e+00
max      2.971473e+00
"""
sns.boxplot(df_final['Proline'])
#As we can see that it is not having any outlier lets check its distribution
sns.histplot(df_final['Proline'],kde=True)
#The featture is now ready for the clustering , it is almost normally distributed with slight right skew

#=======================================================================================
#Phenols
df_final['Phenols'].describe()
"""
mean       0.000000
std        1.002821
min       -2.107246
max        2.539515
"""
sns.boxplot(df_final['Phenols'])
#As we can see that it is not having any outlier lets check its distribution
sns.histplot(df_final['Phenols'],kde=True)
#The featture is now ready for the clustering , it is almost normally distributed 
#=======================================================================================
#Dilution 
df_final['Dilution'].describe()
"""
mean     3.193450e-16
std      1.002821e+00
min     -1.895054e+00
max      1.960915e+00
"""
sns.boxplot(df_final['Dilution'])
#As we can see that it is not having any outlier lets check its distribution
sns.histplot(df_final['Dilution'],kde=True)
#The featture is now ready for the clustering , it is almost normally distributed 

#=====================================================================================
#Proanthocyanins 
df_final['Proanthocyanins'].describe()
"""
mean    -1.197544e-16
std      1.002821e+00
min     -2.069034e+00
max      3.485073e+00
"""
sns.boxplot(df_final['Proanthocyanins'])
#Here we are having two outliers hence we will treat them with winsorinzation 
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Proanthocyanins']
                  )
df_final['Proanthocyanins'] = winsor.fit_transform(df_final[['Proanthocyanins']])
sns.boxplot(df_final['Proanthocyanins'])
#and the outliers have been treated and now we will check the distribution 
sns.histplot(df_final['Proanthocyanins'],kde=True)
#The featture is now ready for the clustering , it is almost normally distributed 
#=========================================================================================
#Flavanoids 
df_final['Flavanoids'].describe()
"""
mean    -3.991813e-16
std      1.002821e+00
min     -1.695971e+00
max      3.062832e+00
"""
sns.boxplot(df_final['Flavanoids'])
#As we can see that it is not having any outlier lets check its distribution
sns.histplot(df_final['Flavanoids'],kde=True)

#=========================================================================================== 
#Hue 
df_final['Hue'].describe()
"""
mean     1.995907e-16
std      1.002821e+00
min     -2.094732e+00
max      3.301694e+00
"""
sns.boxplot(df_final['Hue'])
#Here we are having one outliers hence we will treat them with winsorinzation 
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Hue']
                  )
df_final['Hue'] = winsor.fit_transform(df_final[['Hue']])
sns.boxplot(df_final['Hue'])
#and the outliers have been treated and now we will check the distribution 
sns.histplot(df_final['Hue'],kde=True)
#The featture is now ready for the clustering , it is almost normally distributed 
################################################################################################3

#heirarchical Clustering
from scipy.cluster.hierarchy import linkage 
import scipy.cluster.hierarchy as sch 
#linkage function gives us hierarchical or agglomerative clustering 
#ref the help for linkage 
z=linkage(df_final, method='complete',metric='euclidean') 
plt.figure(figsize=(15,8))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Index')
plt.ylabel('Distance')

#ref help of dendrogram 
#sch.dendrogram(z)
sch.dendrogram(z,leaf_rotation=90,leaf_font_size=10)
plt.show()

from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_final)
#apply labels to the clusters 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_) 
df_final['Clust'] =  cluster_labels
df_final.columns
df1 = df_final.loc[:,['Clust','Alcohol', 'Color', 'Proline', 'Phenols', 'Dilution', 'Proanthocyanins',
       'Flavanoids', 'Hue']]
df1.iloc[:,2:].groupby(df1.Clust).mean()
df1.to_csv('D:/ANACONDA/Data Science/Assignment/Heirarchical1.csv')
#####################################################################################################
#K-Means Clustering 
TWSS = []
k = list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_final)
    
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
df1['Clust'] = mb    
df1.head()  
df1.columns  
df1 = df1.loc[:,['Clust', 'Alcohol', 'Color', 'Proline', 'Phenols', 'Dilution',
       'Proanthocyanins', 'Flavanoids', 'Hue']]    
x = df1.iloc[:,1:].groupby(df1.Clust).mean()  
df1.to_csv('D:/ANACONDA/Data Science/Assignment/kmeans1.csv',encoding='utf-8')

#######################################################################################
#Performing PCA 
df_ = df_final.drop({'Clust'},axis=1)
U,d,Vt = svd(df_)
svd = TruncatedSVD(n_components=3)
svd.fit(df_)
result = pd.DataFrame(svd.transform(df_))
result.head()
result.columns="pc0","pc1","pc2"
result.head()

#Scatter Diagram 

plt.scatter(x=result.pc0, y=result.pc1)






















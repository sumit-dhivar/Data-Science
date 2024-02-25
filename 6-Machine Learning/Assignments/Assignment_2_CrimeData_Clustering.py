# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 22:15:44 2024

@author: sumit Dhivar 

Title:- Agglomerative Clustering
DataSet:- Crime Data 

PS: Perform clustering for the crime data and identify the number of
clusters formed and draw inferences.
"""

##############################################################################
"""
:)Buisness Problem:-
=>Buisness Objective:- 
The primary objective is to identify distinct groupings or clusters within 
the crime data that share similar characteristics.

=>Buisness Constraints:-
The clusters generated should be interpretable and actionable.

"""
"""
Data Dictionary:- 
Feature	                Description	                                       Relevant
Murder	     The number of murders per 100,000 inhabitants.                  Yes
Assault	     The number of assaults per 100,000 inhabitants.	             Yes
UrbanPop	 The percentage of the population living in urban areas.         Yes
Rape	     The number of rapes per 100,000 inhabitants.	                 Yes
"""
################################################################
#Importing Libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

df = pd.read_csv('crime_data.csv')

df.shape
#So this dataset has around 50 tuples and 5 features 
df.columns
# 'Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape' this are the features
#in our dataset 
#According to the dataset Unnamed: 0 feature looks like it is the States of US
# So we will rename this feature for Convenience 
df.rename({'Unnamed: 0': 'State'},axis=1,inplace=True) 

desc = df.describe()
#From this it is seen that the 
#          Murder     Assault   UrbanPop       Rape
# count  50.00000   50.000000  50.000000  50.000000
# mean    7.78800  170.760000  65.540000  21.232000
# std     4.35551   83.337661  14.474763   9.366385
# min     0.80000   45.000000  32.000000   7.300000
# 25%     4.07500  109.000000  54.500000  15.075000
# 50%     7.25000  159.000000  66.000000  20.100000
# 75%    11.25000  249.000000  77.750000  26.175000
# max    17.40000  337.000000  91.000000  46.000000
#-> the average murder rate is around 7.788 in 10000 habitants
#-> the average Assault rate is is 170.76 
# so the murder rate is the lowest and the assault rate is the highest 
#Also the Murder has lowest standard deviation and 
#Assault has the highest std. deviation
#So here we have discused the first and second buisness moment 
#that is finding mean and variance & std. deviation 
df.info() 
#So there are no null values in the data set 

#Pair Plot
sns.pairplot(df)
#From the pair plot we can infer that the feature Assault has 
#moderate positive correlation with Murder 
#While all other show very weak positive correlation


#Let's perform Univariate Analsis
#Murder
from scipy import stats
import pylab
stats.probplot(df['Murder'], dist="norm",plot=pylab)
plt.show()

sns.histplot(data= df['Murder'],kde=True)
#So the Murder feature is slightly right skewed 
#Lets check if it is having any outliers if present then we will treat it 
sns.boxplot(df['Murder'])
#Here no outliers has been observed

#Assault 
stats.probplot(df['Assault'], dist="norm",plot=pylab)
plt.show()
sns.histplot(data= df['Assault'],kde=True)
#So the Assault feautre is bimodal symmetric 
#Lets check if it is having any outliers if present then we will treat it 
sns.boxplot(df['Assault'])
#Here no outliers has been observed

#UrbanPop
stats.probplot(df['UrbanPop'], dist="norm",plot=pylab)
plt.show()
sns.histplot(data= df['UrbanPop'],kde=True)
sns.boxplot(df['UrbanPop'])
#So this feature is also normal and no outliers are there 

#Rape 
stats.probplot(df['UrbanPop'], dist="norm",plot=pylab)
plt.show()
sns.histplot(data= df['UrbanPop'],kde=True)
#Slightly left skewed
sns.boxplot(df['UrbanPop'])
#No outliers are there 

#Here the state are nominal data type hence we will not use it in clustering
#as it may result in wrong clustering for agglomerative
df.drop({'State'},axis=1,inplace=True) 
##################################################################3
#Let's now apply this data for Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df)
#apply labels to the clusters 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
df['Clust'] = cluster_labels 

df = df.iloc[:,[4,0,1,2,3]]
df.iloc[:,1:].groupby(df.Clust).mean()

# So to see which state comes in which cluster
df_new = pd.read_csv('crime_data.csv')
df['State'] = df_new['Unnamed: 0']
df = df.iloc[:,[5,0,1,2,3,4]]

df.head(5)
#So we can see that which state comes under which category and what are the parameters








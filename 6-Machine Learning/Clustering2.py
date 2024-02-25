# -*- coding: utf-8 -*-
"""
Created on Tue Oct 17 09:17:56 2023

@author: sumit
"""

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
#Let us try to understand first how k means works for 2-D data 
#for taht, generate random numbers in the range 0 to 1 
# and with uniform probability of 1/50
X = np.random.uniform(0,1,50)
Y = np.random.uniform(0,1,50)

#Create a emppty data frame with 0 rows and 2 columns 
df_xy = pd.DataFrame(columns=["X","Y"])
#assign the values of X and Y to these columns 
df_xy.X = X 
df_xy.Y = Y 
df_xy.plot(x='X',y='Y', kind='scatter')
model1=KMeans(n_clusters=3).fit(df_xy)
"""
With data X and Y, apply Kmeans model, 
generate scatter plot
with scale / font = 10
cmap = plt.cm.coolwarm:cool color combination
"""

model1.labels_ 
df_xy.plot(x='X',y='Y',c=model1.labels_,kind='scatter',s=10,
           cmap = plt.cm.coolwarm)

Univ1 = pd.read_excel('University_Clustering.xlsx')

Univ1.describe()
Univ = Univ1.drop(['State'],axis=1)

#we that there is scale difference among the columns wich we have either by using normalization or standaedization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x 
#Now apply this normalization function to Univ dataframe for all the row and columns from 1 till end
#since 0th column has university name hence skipped 
df_norm = norm_func(Univ.iloc[:,1:])
#What will be ideal cluster number, will it be 1,2,3
TWSS = []
k = list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    
    TWSS.append(kmeans.inertia_)#total within sum of square 
    
'''
KMeans inertia, also 
'''
    
TWSS
#As k value increases the TWSS value decreases 
plt.plot(k,TWSS,'ro-');
plt.xlabel('No_of_clusters');
plt.ylabel('Total_within_SS');

'''
How to select value of k from elbow curve
When k changes from 2 to 3, then decreases
in twss is higher than 
When k changes from 3 to 4.
When k values changes from 5 to 6 decreases
in twss is considerably less, hence considered k = 3
''' 
    
model = KMeans(n_clusters=3)    
model.fit(df_norm)    
model.labels_ #This shows that the data point is in which cluster   
mb = pd.Series(model.labels_)    
Univ['clust'] = mb    
Univ.head()    
Univ = Univ.iloc[:,[7,0,1,2,3,4,5,6]]    
Univ    
x = Univ.iloc[:,2:8].groupby(Univ.clust).mean()    

Univ.to_csv('kmeans_University.csv',encoding='utf-8')
import os 
os.getcwd()    



    
    
    
    
    
    
    
    
    
    
    
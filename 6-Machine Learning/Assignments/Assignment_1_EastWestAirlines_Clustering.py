# -*- coding: utf-8 -*-
"""
Created on Oct 12 20:34:12 2023

@author: sumit

Title:- Agglomerative Clustering
DataSet:- EastWest AirLines
"""
##############################################################################
"""
:)Buisness Problem:-
=>Buisness Objective:- 
1. Cluster customers based on their behaviours such as frequency of travel, 
destinations, booking preferences, loyalty program engagement, etc., the airline 
can gain insights into different customer segments, So it can be used to tailor 
marketing strategies, services.
2. Offering promotions or services that are most relevant to each cluster/group.
3. By identifying high-value customer segments, the airline can prioritize 
efforts to retain and attract these customers, potentially increasing revenue.

=>Buisness Constraints:-
1. Missing or inaccurate data could lead to biased clustering results and 
unreliable insights.
2. Performing clustering analysis may require significant computational resources,
 especially for large datasets or complex algorithms.

"""
###############################################################################
"""
Data Dictionary:- 

Feature	                     Description	                                                Relevant?
ID#	               Unique identifier for each customer -                              	        No
Balance         	Number of miles remaining in the customer's account                         Yes
Qual_miles	        Number of miles qualifying for frequent flyer status	                    Yes
cc1_miles	        Number of miles earned with airline for credit card 1	                    Yes
cc2_miles	        Number of miles earned with airline for credit card 2	                    Yes
cc3_miles	        Number of miles earned with airline for credit card 3                     	Yes
Bonus_miles     	Number of bonus miles earned by customer	                                Yes
Bonus_trans	        Number of transactions on the account earning bonus miles	                Yes
Flight_miles_12mo	Number of flight miles flown in the past 12 months	                        Yes
Flight_trans_12 	Number of flight transactions in the past 12 months                        	Yes
Days_since_enroll	Number of days since the customer enrolled in the frequent flyer program	Yes
Award?	            Indicates whether the customer received an award flight	                    Yes

"""

################################################################
#Importing Libraries
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

################################################################
#-------------------------EDA-----------------------------------
df = pd.read_excel('EastWestAirlines.xlsx')

df.columns
# 'ID#', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
# 'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
# 'Days_since_enroll', 'Award?'
#Here we can see that some features name is not according to the convention
#So we will rename them if they are relevant or else we will drop them 

df.drop({'ID#'},axis=1,inplace=True)
#Here we have dropped the ID# feature as it does not give valuable information

df.rename({'Award?': 'Award'},axis=1,inplace=True)
#Here we have renamed the Award feature.

df.info()
#all the features are of integer data type 

desc = df.describe()
#here the standard deviation of some feature is high while some are very low

#Also the data is scaled differently, hence we will apply normalisation to the
# dataset and convert it into same range 
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x 

df_norm = norm_func(df.iloc[:,:])

desc = df_norm.describe()
#Now we can find the outliers in the data set 

plt.rcParams['figure.figsize'] = (18,6)
sns.boxplot(data=df_norm)
#Only three features don't have any outlier i.e cc1_miles, Days_since_enroll, Award
###############################################################
#Let's Perform Univariate Analysis
#1. Balance
sns.scatterplot(data = df_norm['Balance'])
sns.histplot(data=df_norm['Balance'],kde=True)
#So the Balance feature is having very high variance and also has outliers 
#So we will treat Balance feature with winsorization to make it free from outliers
from feature_engine.outliers import Winsorizer
#For the feature Balance
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Balance']
                  )
df_norm['Balance'] = winsor.fit_transform(df_norm[['Balance']])
#Outliers from the Balance feature has been removed
sns.scatterplot(data = df_norm['Balance'])
sns.histplot(data=df_norm['Balance'],kde=True)
#------------------------------------------------------------
# 2. Qual_miles feature
sns.scatterplot(data = df_norm['Qual_miles'])
sns.histplot(data=df_norm['Qual_miles'],kde=True)
#So this feature has a very low variance and does not 
#give any important information so we will drop it 
df_norm.drop({'Qual_miles'},axis=1,inplace=True)
#------------------------------------------------------------
# 3. cc1_miles 
sns.scatterplot(df_norm['cc1_miles'])
sns.histplot(df_norm['cc1_miles'],kde=True)
#So this feature has a very low variance and does not 
#give any important information so we will drop it 
df_norm.drop({'cc1_miles'},axis=1,inplace=True)
#------------------------------------------------------------
# 4. cc2 and cc3 also have same i.e low variance hence we 
#will drop them 
df_norm.drop({'cc2_miles'},axis=1,inplace=True)
df_norm.drop({'cc3_miles'},axis=1,inplace=True)
#------------------------------------------------------------
# 5. Bonus_miles
sns.scatterplot(data = df_norm['Bonus_miles'])
sns.histplot(data=df_norm['Bonus_miles'],kde=True)
#It is right skewed and has outliers
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Bonus_miles']
                  )
df_norm['Bonus_miles'] = winsor.fit_transform(df_norm[['Bonus_miles']])
sns.boxplot(data=df_norm)
#Hence the outliers has been removed
#--------------------------------------------------------------
#6. Bonus_trans 
sns.scatterplot(data = df_norm['Bonus_trans'])
sns.histplot(data=df_norm['Bonus_trans'],kde=True)
#So this feature is right skewed and has outliers so we will 
#treat this outlier with winsorization 
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Bonus_trans']
                  )
df_norm['Bonus_trans'] = winsor.fit_transform(df_norm[['Bonus_trans']])
sns.boxplot(data=df_norm)
#---------------------------------------------------------------------------
#7. Flight_miles_12mo
sns.scatterplot(data = df_norm['Flight_miles_12mo'])
sns.histplot(data=df_norm['Flight_miles_12mo'],kde=True)
df_norm['Flight_miles_12mo'].describe()
#So we can see that the standard deviation and the variance is very low 
#Hence we will drop thhis feature 
df_norm.drop({'Flight_miles_12mo'},axis=1,inplace=True)
#----------------------------------------------------------------------------
#8. Flight_trans_12
sns.scatterplot(data = df_norm['Flight_trans_12'])
sns.histplot(data=df_norm['Flight_trans_12'],kde=True)
df_norm['Flight_trans_12'].describe()
#So we can see that the standard deviation and the variance is very low 
#Hence we will drop thhis feature 
df_norm.drop({'Flight_trans_12'},axis=1,inplace=True)
#----------------------------------------------------------------------------
sns.boxplot(data=df_norm)
df_norm[['Days_since_enroll','Award']].describe()

sns.histplot(df_norm['Days_since_enroll'],kde=True)
sns.histplot(df_norm['Award'],kde=True)

sns.scatterplot(data = df_norm['Days_since_enroll'])
sns.scatterplot(data = df_norm['Award'])
# Therefore the Award is a binary feature 0 and 1 
#Let's us drop it for more good result 
df_norm.drop({'Award'},axis=1,inplace=True)
#----------------------------------------------------------------------------
#Let's now apply this data for Agglomerative Clustering
from sklearn.cluster import AgglomerativeClustering 
h_complete = AgglomerativeClustering(n_clusters=3,linkage='complete',affinity='euclidean').fit(df_norm)
#apply labels to the clusters 
h_complete.labels_
cluster_labels = pd.Series(h_complete.labels_)
df_norm['Clust'] = cluster_labels 

df_norm = df_norm.iloc[:,[4,0,1,2,3]]
df_norm.iloc[:,1:].groupby(df_norm.Clust).mean()




# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:27:07 2024

@author: sumit
"""

import pandas as pd
import numpy as np
df =pd.read_csv("Avacado_Price.csv")
df.columns
#explotatary data analysis and data preprocessing
type(df)
df.shape
df.dtypes
df.describe()
df.info()
df.std()
df.var()

df.isna().sum()#no null values
df.duplicated().sum()#no dulicated values

#Graphical represpentation
import matplotlib.pyplot as plt
#Histogram
plt.hist(df['AveragePrice'])
plt.hist(df['Total_Volume'])
plt.hist(df["tot_ava1"])
plt.hist(df['tot_ava2'])
plt.hist(df['tot_ava3'])
plt.hist(df['Total_Bags'])
plt.hist(df["Small_Bags"])
plt.hist(df['Large_Bags'])
plt.hist(df['XLarge Bags'])

#boxplot
plt.boxplot(df['AveragePrice'])
plt.boxplot(df['Total_Volume'])
plt.boxplot(df["tot_ava1"])
plt.boxplot(df['tot_ava2'])
plt.boxplot(df['tot_ava3'])
plt.boxplot(df['Total_Bags'])
plt.boxplot(df["Small_Bags"])
plt.boxplot(df['Large_Bags'])
plt.boxplot(df['XLarge Bags'])
#almost all columns as outliers but we can't adjust because data itself in its way we can only do outlier treatment on client word.

#scatter plots
plt.scatter(df['Total_Volume'],df['AveragePrice'])
plt.scatter(df['tot_ava1'],df['AveragePrice'])
plt.scatter(df["tot_ava2"],df['AveragePrice'])
plt.scatter(df['tot_ava3'],df['AveragePrice'])
plt.scatter(df['Total_Bags'],df['AveragePrice'])
plt.scatter(df["Small_Bags"],df['AveragePrice'])
plt.scatter(df['Large_Bags'],df['AveragePrice'])
plt.scatter(df["XLarge Bags"],df['AveragePrice'])

# Jointplot
import seaborn as sns
sns.jointplot(df['Total_Volume'],df['AveragePrice'])
sns.jointplot(df['tot_ava1'],df['AveragePrice'])
sns.jointplot(df["tot_ava2"],df['AveragePrice'])
sns.jointplot(df['tot_ava3'],df['AveragePrice'])
sns.jointplot(df['Total_Bags'],df['AveragePrice'])
sns.jointplot(df["Small_Bags"],df['AveragePrice'])
sns.jointplot(df['Large_Bags'],df['AveragePrice'])
sns.jointplot(df['XLarge Bags'],df['AveragePrice'])

sns.pairplot(df)
df.corr()

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder() #factorise the data
df["year"] = lb.fit_transform(df["year"])

df_new= pd.get_dummies(df,drop_first=True)
df_new=df_new.rename(columns={'XLarge Bags':'XLarge_Bags'})
df_new.columns

#preparing the model
import statsmodels.formula.api as smf
f1='AveragePrice ~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+year+type_organic+region_Atlanta+region_BaltimoreWashington+region_Boise+region_Boston+region_BuffaloRochester+region_California+region_Charlotte+region_Chicago+region_CincinnatiDayton+region_Columbus+region_DallasFtWorth+region_Denver+region_Detroit+region_GrandRapids+region_GreatLakes+region_HarrisburgScranton+region_HartfordSpringfield+region_Houston+region_Indianapolis+region_Jacksonville+region_LasVegas+region_LosAngeles+region_Louisville+region_MiamiFtLauderdale+region_Midsouth+region_Nashville+region_NewOrleansMobile+region_NewYork+region_Northeast+region_NorthernNewEngland+region_Orlando+region_Philadelphia+region_PhoenixTucson+region_Pittsburgh+region_Plains+region_Portland+region_RaleighGreensboro+region_RichmondNorfolk+region_Roanoke+region_Sacramento+region_SanDiego+region_SanFrancisco+region_Seattle+region_SouthCarolina+region_SouthCentral+region_Southeast+region_Spokane+region_StLouis+region_Syracuse+region_Tampa+region_TotalUS+region_West+region_WestTexNewMexico'
model1=smf.ols(formula=f1,data=df_new).fit()
model1.summary()

import statsmodels.api as sm
sm.graphics.influence_plot(model1)

data_new=df_new.drop(df_new.index[[]])
#prepare model
model_new=smf.ols(f1,data=data_new).fit()
model_new.summary()
sns.pairplot(data_new)
data_new.corr()
#p_values for more than 0.05 r^2 =0.96

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_tv = smf.ols('Total_Volume ~ tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+year+type_organic+region_Atlanta+region_BaltimoreWashington+region_Boise+region_Boston+region_BuffaloRochester+region_California+region_Charlotte+region_Chicago+region_CincinnatiDayton+region_Columbus+region_DallasFtWorth+region_Denver+region_Detroit+region_GrandRapids+region_GreatLakes+region_HarrisburgScranton+region_HartfordSpringfield+region_Houston+region_Indianapolis+region_Jacksonville+region_LasVegas+region_LosAngeles+region_Louisville+region_MiamiFtLauderdale+region_Midsouth+region_Nashville+region_NewOrleansMobile+region_NewYork+region_Northeast+region_NorthernNewEngland+region_Orlando+region_Philadelphia+region_PhoenixTucson+region_Pittsburgh+region_Plains+region_Portland+region_RaleighGreensboro+region_RichmondNorfolk+region_Roanoke+region_Sacramento+region_SanDiego+region_SanFrancisco+region_Seattle+region_SouthCarolina+region_SouthCentral+region_Southeast+region_Spokane+region_StLouis+region_Syracuse+region_Tampa+region_TotalUS+region_West+region_WestTexNewMexico', data = df_new).fit().rsquared  
vif_tv = 1/(1 - rsq_tv) 
rsq_ta1 = smf.ols('tot_ava1 ~  Total_Volume+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+year+type_organic+region_Atlanta+region_BaltimoreWashington+region_Boise+region_Boston+region_BuffaloRochester+region_California+region_Charlotte+region_Chicago+region_CincinnatiDayton+region_Columbus+region_DallasFtWorth+region_Denver+region_Detroit+region_GrandRapids+region_GreatLakes+region_HarrisburgScranton+region_HartfordSpringfield+region_Houston+region_Indianapolis+region_Jacksonville+region_LasVegas+region_LosAngeles+region_Louisville+region_MiamiFtLauderdale+region_Midsouth+region_Nashville+region_NewOrleansMobile+region_NewYork+region_Northeast+region_NorthernNewEngland+region_Orlando+region_Philadelphia+region_PhoenixTucson+region_Pittsburgh+region_Plains+region_Portland+region_RaleighGreensboro+region_RichmondNorfolk+region_Roanoke+region_Sacramento+region_SanDiego+region_SanFrancisco+region_Seattle+region_SouthCarolina+region_SouthCentral+region_Southeast+region_Spokane+region_StLouis+region_Syracuse+region_Tampa+region_TotalUS+region_West+region_WestTexNewMexico', data = df_new).fit().rsquared  
vif_ta1 = 1/(1 - rsq_ta1) 
rsq_ta2 = smf.ols('tot_ava2 ~ Total_Volume+tot_ava1+tot_ava3+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+year+type_organic+region_Atlanta+region_BaltimoreWashington+region_Boise+region_Boston+region_BuffaloRochester+region_California+region_Charlotte+region_Chicago+region_CincinnatiDayton+region_Columbus+region_DallasFtWorth+region_Denver+region_Detroit+region_GrandRapids+region_GreatLakes+region_HarrisburgScranton+region_HartfordSpringfield+region_Houston+region_Indianapolis+region_Jacksonville+region_LasVegas+region_LosAngeles+region_Louisville+region_MiamiFtLauderdale+region_Midsouth+region_Nashville+region_NewOrleansMobile+region_NewYork+region_Northeast+region_NorthernNewEngland+region_Orlando+region_Philadelphia+region_PhoenixTucson+region_Pittsburgh+region_Plains+region_Portland+region_RaleighGreensboro+region_RichmondNorfolk+region_Roanoke+region_Sacramento+region_SanDiego+region_SanFrancisco+region_Seattle+region_SouthCarolina+region_SouthCentral+region_Southeast+region_Spokane+region_StLouis+region_Syracuse+region_Tampa+region_TotalUS+region_West+region_WestTexNewMexico', data = df_new).fit().rsquared  
vif_ta2 = 1/(1 - rsq_ta2)
rsq_ta3 = smf.ols('tot_ava3 ~Total_Volume+tot_ava1+tot_ava2+Total_Bags+Small_Bags+Large_Bags+XLarge_Bags+year+type_organic+region_Atlanta+region_BaltimoreWashington+region_Boise+region_Boston+region_BuffaloRochester+region_California+region_Charlotte+region_Chicago+region_CincinnatiDayton+region_Columbus+region_DallasFtWorth+region_Denver+region_Detroit+region_GrandRapids+region_GreatLakes+region_HarrisburgScranton+region_HartfordSpringfield+region_Houston+region_Indianapolis+region_Jacksonville+region_LasVegas+region_LosAngeles+region_Louisville+region_MiamiFtLauderdale+region_Midsouth+region_Nashville+region_NewOrleansMobile+region_NewYork+region_Northeast+region_NorthernNewEngland+region_Orlando+region_Philadelphia+region_PhoenixTucson+region_Pittsburgh+region_Plains+region_Portland+region_RaleighGreensboro+region_RichmondNorfolk+region_Roanoke+region_Sacramento+region_SanDiego+region_SanFrancisco+region_Seattle+region_SouthCarolina+region_SouthCentral+region_Southeast+region_Spokane+region_StLouis+region_Syracuse+region_Tampa+region_TotalUS+region_West+region_WestTexNewMexico', data = df_new).fit().rsquared  
vif_ta3 = 1/(1 - rsq_ta3)
rsq_tb = smf.ols('Total_Bags ~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+Small_Bags+Large_Bags+XLarge_Bags+year+type_organic+region_Atlanta+region_BaltimoreWashington+region_Boise+region_Boston+region_BuffaloRochester+region_California+region_Charlotte+region_Chicago+region_CincinnatiDayton+region_Columbus+region_DallasFtWorth+region_Denver+region_Detroit+region_GrandRapids+region_GreatLakes+region_HarrisburgScranton+region_HartfordSpringfield+region_Houston+region_Indianapolis+region_Jacksonville+region_LasVegas+region_LosAngeles+region_Louisville+region_MiamiFtLauderdale+region_Midsouth+region_Nashville+region_NewOrleansMobile+region_NewYork+region_Northeast+region_NorthernNewEngland+region_Orlando+region_Philadelphia+region_PhoenixTucson+region_Pittsburgh+region_Plains+region_Portland+region_RaleighGreensboro+region_RichmondNorfolk+region_Roanoke+region_Sacramento+region_SanDiego+region_SanFrancisco+region_Seattle+region_SouthCarolina+region_SouthCentral+region_Southeast+region_Spokane+region_StLouis+region_Syracuse+region_Tampa+region_TotalUS+region_West+region_WestTexNewMexico', data = df_new).fit().rsquared  
vif_tb = 1/(1 - rsq_tb)
rsq_sb = smf.ols('Small_Bags ~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Large_Bags+XLarge_Bags+year+type_organic+region_Atlanta+region_BaltimoreWashington+region_Boise+region_Boston+region_BuffaloRochester+region_California+region_Charlotte+region_Chicago+region_CincinnatiDayton+region_Columbus+region_DallasFtWorth+region_Denver+region_Detroit+region_GrandRapids+region_GreatLakes+region_HarrisburgScranton+region_HartfordSpringfield+region_Houston+region_Indianapolis+region_Jacksonville+region_LasVegas+region_LosAngeles+region_Louisville+region_MiamiFtLauderdale+region_Midsouth+region_Nashville+region_NewOrleansMobile+region_NewYork+region_Northeast+region_NorthernNewEngland+region_Orlando+region_Philadelphia+region_PhoenixTucson+region_Pittsburgh+region_Plains+region_Portland+region_RaleighGreensboro+region_RichmondNorfolk+region_Roanoke+region_Sacramento+region_SanDiego+region_SanFrancisco+region_Seattle+region_SouthCarolina+region_SouthCentral+region_Southeast+region_Spokane+region_StLouis+region_Syracuse+region_Tampa+region_TotalUS+region_West+region_WestTexNewMexico', data = df_new).fit().rsquared  
vif_sb = 1/(1 - rsq_sb)
rsq_lb = smf.ols('Large_Bags ~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+XLarge_Bags+year+type_organic+region_Atlanta+region_BaltimoreWashington+region_Boise+region_Boston+region_BuffaloRochester+region_California+region_Charlotte+region_Chicago+region_CincinnatiDayton+region_Columbus+region_DallasFtWorth+region_Denver+region_Detroit+region_GrandRapids+region_GreatLakes+region_HarrisburgScranton+region_HartfordSpringfield+region_Houston+region_Indianapolis+region_Jacksonville+region_LasVegas+region_LosAngeles+region_Louisville+region_MiamiFtLauderdale+region_Midsouth+region_Nashville+region_NewOrleansMobile+region_NewYork+region_Northeast+region_NorthernNewEngland+region_Orlando+region_Philadelphia+region_PhoenixTucson+region_Pittsburgh+region_Plains+region_Portland+region_RaleighGreensboro+region_RichmondNorfolk+region_Roanoke+region_Sacramento+region_SanDiego+region_SanFrancisco+region_Seattle+region_SouthCarolina+region_SouthCentral+region_Southeast+region_Spokane+region_StLouis+region_Syracuse+region_Tampa+region_TotalUS+region_West+region_WestTexNewMexico', data = df_new).fit().rsquared  
vif_lb = 1/(1 - rsq_lb)
rsq_xlb = smf.ols('XLarge_Bags ~ Total_Volume+tot_ava1+tot_ava2+tot_ava3+Total_Bags+Small_Bags+Large_Bags+year+type_organic+region_Atlanta+region_BaltimoreWashington+region_Boise+region_Boston+region_BuffaloRochester+region_California+region_Charlotte+region_Chicago+region_CincinnatiDayton+region_Columbus+region_DallasFtWorth+region_Denver+region_Detroit+region_GrandRapids+region_GreatLakes+region_HarrisburgScranton+region_HartfordSpringfield+region_Houston+region_Indianapolis+region_Jacksonville+region_LasVegas+region_LosAngeles+region_Louisville+region_MiamiFtLauderdale+region_Midsouth+region_Nashville+region_NewOrleansMobile+region_NewYork+region_Northeast+region_NorthernNewEngland+region_Orlando+region_Philadelphia+region_PhoenixTucson+region_Pittsburgh+region_Plains+region_Portland+region_RaleighGreensboro+region_RichmondNorfolk+region_Roanoke+region_Sacramento+region_SanDiego+region_SanFrancisco+region_Seattle+region_SouthCarolina+region_SouthCentral+region_Southeast+region_Spokane+region_StLouis+region_Syracuse+region_Tampa+region_TotalUS+region_West+region_WestTexNewMexico', data = df_new).fit().rsquared  
vif_xlb = 1/(1 - rsq_xlb)

d1={"variables":["total_vol","ava1","ava2","ava3","tot_bag","small","large","xlarge"],"vif":[vif_tv,vif_ta1,vif_ta2,vif_ta3,vif_tb,vif_sb,vif_lb,vif_xlb]}
dataframe=pd.DataFrame(d1)
dataframe  #all values are less than 10

#so final model
### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
df_train, df_test = train_test_split(df_new, test_size = 0.2,random_state=2) # 20% test data
# preparing the model on train data 
model_train = smf.ols(f1, data = df_train).fit()

# prediction on test data set 
test_pred = model_train.predict(df_test)
test_resid = test_pred - df_test.AveragePrice
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse #0.26

# train_data prediction
train_pred = model_train.predict(df_train)
train_resid  = train_pred - df_train.AveragePrice
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse #0.26

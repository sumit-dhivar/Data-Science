# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:32:19 2024

@author: sumit
"""

import pandas as pd 
import matplotlib.pyplot as plt 
import numpy as np
import statsmodels.formula.api as sm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report
claimants = pd.read_csv('claimants.csv')
#There are CLMAGE and LOSS are having continous dataa rest are discrete
#verify the dataset, where CASENUM is not really useful so dropping it will be good
c1 = claimants.drop('CASENUM' , axis = 1)
c1.head(11)
c1.describe()
#Let us check whether their are NULL values
c1.isna().sum()
#There are several null values
# if we will use dropna() function we will lose almost 290 datapoints
# hence we will go for imputation
c1.dtypes
mean_value = c1.CLMAGE.mean()
mean_value
#Now let us impute the same
c1.CLMAGE = c1.CLMAGE.fillna(mean_value)
c1.CLMAGE.isna().sum()
# hence all null values of CLMAGE has been filled by mena value
# for columns where there are discrete values, we will apply mode imputation
mode_CLMSEX = c1.CLMSEX.mode()
mode_CLMSEX
c1.CLMSEX = c1.CLMSEX.fillna((mode_CLMSEX)[0])
c1.CLMSEX.isna().sum()
# CLMINSUR is also categorical data hence mode imputation is applied
mode_CLMINSUR = c1.CLMINSUR.mode()
mode_CLMINSUR
c1.CLMINSUR = c1.CLMINSUR.fillna((mode_CLMINSUR)[0])
c1.CLMINSUR.isna().sum()
#SEATBELT is categorical data hence go for model imputation
mode_SEATBELT = c1.SEATBELT.mode()
mode_SEATBELT 
c1.SEATBELT = c1.SEATBELT.fillna((mode_SEATBELT)[0])
c1.SEATBELT.isna().sum()
# Now the person we meet an accident will hire the atterney or not
# Let us build the model
logit_model = sm.logit('ATTORNEY ~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data=c1).fit()
logit_model.summary()
#in logistic regression we do not have R squared values, only check p-values 
#SEATBELT is statistically insignificant ignore and proceed
logit_model.summary2()
#here we are goint to check AIC vlaue, it stands for Akaike information 
#is mathematical method for evaluation how well a model fits the data 
#A lower the score more the better model, AIC scoress are only useful in 
#with other AIC scores for the same 

#Now let us go for predictions 
pred = logit_model.predict(c1.iloc[:,1:])
# here we are applying all rows columns from 1,  as column 0 is ATTORNEY 
#target value 
#let us check the performance of the model 
fpr, tpr, thresholds = roc_curve(c1.ATTORNEY, pred)
#we are appluing actual values and predicted values so as to get 
#false positive rate, true positive reate and threshold 
#the optimal Cutoff value is the point where there is high true positive 
#you can use the below code to get the value: 
optimal_idx = np.argmax(tpr-fpr)
optimal_threshold = thresholds [optimal_idx]
optimal_threshold 
#ROC:receiver operating charecteristics curve in logistic regression are determining 
#best cuttoff/threshold value 
import pylab as pl
i = np.arange(len(tpr))# index for df
# here tpr is of 559 so it will create a scale of 0 to 558
roc = pd.DataFrame({'fpr' : pd.Series(fpr , index = i),
                    'tpr' : pd.Series(tpr , index = i),
                    '1-fpr' : pd.Series(1-fpr , index = i),
                    'tf' : pd.Series(tpr - (1-fpr) , index = i ),
                    'thresholds' : pd.Series(thresholds , index = i)})
#we want to create a dataframe which comprises of columns fpr, 
#tpr, 1-fpr, tpr - (1-fpr)
#The optimal cut off would be where tpr is high and fpr is low 
# tpr - (1-fpr) is zero or neat to zero is the optimal cutoff point 
#plot ROC curve 
plt.plot(fpr.tpr)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate ")
roc.iloc[(roc.tf-0).abs().argsort()[:1]]
roc_auc = auc(fpr,tpr)
print("Area under the curve:%f"%roc_auc)
#Area is 0.7601 

#tpr vs 1-fpr 
#Plot tpr vs 1-fpr 
fig, ax =  pl.subplots()
pl.plots(roc['tpr'], color='red')
pl.plots(roc['1-fpr'], color='blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receive operating charecteristic')
ax.set_xticklabels([])
#The optimal cut off point is 0.3176
#So anything above this can be labelled as 1 else 0 
#You can see from the output/chart that where TPR is crossing 1-FPR th
#fpr is 36% and tpr-(1-fpr) is nearest to zero
#in the current example 
#filling all the cells with zeros 
c1['pred'] = np.zeros(1340)
c1.loc[pred>optimal_threshold,"pred"]=1
#let us check the classification report 
classification = classification_report(c1['pred'],c1['ATTORNEY'])
classification

#splitting the data into train and test 
train_data = test_data = train_test_split(c1, test_size=0.3)
#Model building 
model = sm.logit('ATTORNEY ~CLMAGE+LOSS+CLMINSUR+CLMSEX+SEATBELT',data=train_data).fit()
model.summary()
#p values are below the condition of 0.05 
#but SEATBELT has got statistically insignificant
model.summary2()
#AIC value is 1110.3782 , AIC score useful in comparison with other 
#lower the aic score better the model 
#let us us go for prediction 








































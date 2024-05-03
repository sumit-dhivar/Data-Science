# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 09:30:51 2024

@author: sumit
"""

import pandas as pd 
import numpy as np
import scipy
from scipy import stats

import statsmodels.stats.descriptivestats as sd
#from statsmodels.stats import Weightstats as stats 
import statsmodels.stats.weightstats as stests
#1 sample sign test 
#for given dataset check whether scores are qwual or less than 50 
#H0 = scores are either equal or less than 80 
#H1 = scoress are not equal n greater than 80 
#Whenever there is single sample and data is not normal
marks = pd.read_csv('Signtest.csv')
#Normal QQ Plot 
import pylab
stats.probplot(marks.Scores, dist='norm',plot=pylab)
#Data is not normal 
#H0 - data is normal 
#H1 -data is not normal 
stats.shapiro(marks.Scores)
#p_value is 0.024299481883645058, p is low null go 
#p_value is 0.024299481883645058 > 0.005 , p is high null fly/apply
#Decision :  data is not normal
#H1 is valid normal
###########################################
#Let us check the distribution of the data 
marks.Scores.describe()
#1-sample sign test 
sd.sign_test(marks.Scores,mu0=marks.Scores.mean())
#p_ value is 0.02> 0.05  so p is high null fly 
#Decision :
#H0 scores one either equal or less than 80




#############1 - Sample Z-Test ##########################
#importing the data 
fabric = pd.read_csv('Fabric_data.csv')
#calculating the normality test 
print(stats.shapiro(fabric))
#0.1460>0.005 H0 True 
#Calculating the mean 
np.mean(fabric)

#ztest 
# parameters in z tests, value is mean of data 
ztest, pval = stests.ztest(fabric,x2=None, value=150)

print(float(pval))

#p_value = 7.156e-06 < 0.05 so p low null go

#############################################################
##############------Mann- Whitney test-------################
#Vehicle with and without additives 
#H0 fuel additive does not impact the performance 
#H1 : fuel additive impacts the performance 
fuel = pd.read_csv('mann_whitney_additive.csv')
fuel

fuel.columns = "Without_additive", "With_additive"
#Normality test 
# H0 : data is normal 
print(stats.shapiro(fuel.Without_additive))
print(stats.shapiro(fuel.With_additive))
#Without_additive is normal 
#With additive is not normal 
#When two samples are not normal then mannwhitney test 
#Non - Parametric test case 
#Mann- Whitney test 
scipy.stats.mannwhiteyu(fuel.Without_additive,fuel.With_additve)
#p_value = 0.4437 > 0.05 so p high null fly 
#H0 : fuel additive does not impact the performance

###################################################################### 
##############------Paired t - test-----####################
#WHen two features are normal then paird T test
#A univariate test that tests for a significant difference between  
sup = pd.read_csv("paired2.csv") 
#H0 : There is no significant differnece between means of suppliers 
#Ha : There is significant differenece between means of suppliers 
#Normality Test - # Shapiro Test 
stats.shapiro(sup.SupplierA)
stats.shapiro(sup.SupplierB)
#Data are Normal 
import seaborn as sns 
sns.boxplot(data = sup)

#Assuming the external Conditions are same for both the samples 
#Paired T-Tests 
ttest, pval = stats.ttest_rel(sup['SupplierA'], sup['SupplierB'])
print(pval)

#p-value is low null go 


##########################################################################

################### - ---- 2-Sample T-test ----------#####################

#Load Data 
prom = pd.read_excel('Promotion.xlsx')
prom
#H0 : InterestRateWaiver < StandardPromotion 
#Ha : InterestRateWaiver > StandardPromotion  
prom.columns = "InterestRateWaiver", "StandardPromotion"
#Normality Test
stats.shapiro(prom.InterestRateWaiver)
print(stats.shapiro(prom.StandardPromotion))
#Data is normal  
#Variance test 
print(scipy.stats.levene)
#Ho: Both columns have equal variance 
#Ha: Both columns have unequal variance 
scipy.stats.levene(prom.InterestRateWaiver, prom.StandardPromotion)
#p-value = 0.287 > 0.05 so p high null fly => zequal variances

# 2 Sample t test 
scipy.stats.ttest_ind(prom.InterestRateWaiver,prom.StandardPromotion)
help(scipy.stats.ttest_ind)

######################################################################## 
##########-----------One-Way ANOVA---------##################

con_renewal = pd.read_excel('ContractRenewal_Data(unstacked).xlsx')
con_renewal.columns = "SupplierA" ,"SupplierB", "SupplierC"
#H0 : All the 3 suppliers have equal mean transaction time 
#Ha :All the 3 suppliers have not equal mean transaction time 
#Normality test 
stats.shapiro(con_renewal.SupplierA)
#p-value = 0.89>0.05 SuppplierA is normal 
stats.shapiro(con_renewal.SupplierB)
#pvalue=0.648 
stats.shapiro(con_renewal.SupplierC)
# pvalue=0.57 >0.05 SupplierC is normal 
#Variancce test 
help(scipy.stats.levene)
#All 3 suppliers are being checked for variances 
scipy.stats.levene(con_renewal.SupplierA,con_renewal.SupplierB,con_renewal.SupplierC)
#The levene test tests the null hypothesis 
#That all the inouts samples are from popuulation wiht equal variances 
#pvalue = 0.777 > 0.05, p is high null fly 
# H0 = All inpputs samples are from populations with equal variances 

#One - way anova 
F, p = stats.f_oneway(con_renewal.SupplierA,con_renewal.SupplierB,con_renewal.SupplierC)

#p-Value 
#P high null fly 

#All the 3 supplier have equall mean transaction time 

################ 2-Proportion test 
import numpy as np 
two_prop_test  = pd.read_excel('JohnyTalkers.xlsx')
from statsmodels.stats.proportion import proportions_ztest
tab1 = two_prop_test.Person.value_counts()
tab1
tab2 = two_prop_test.Drinks.value_counts()
tab2

#crosstable table 
pd.crosstab(two_prop_test.Person, two_prop_test.Drinks)

count = np.array([58,152])
nobs = np.array([480,740])

stats, pval = proportions_ztest(count, nobs, alternative='two-sided')
print(pval)#Pavalue = 0.00013
stats, pval = proportions_ztest(count, nobs, alternative='larger')
print(pval)

####################################################################
###################------Chi Squared Test------#####################
Bahaman = pd.read_excel('Bahaman.xlsx')
Bahaman

count =  pd.crosstab(Bahaman['Defective'], Bahaman['Country'])
count

Chisquares_results = scipy.stats.chi2_contingency(count)
Chi_square = [['Test Statistic', 'p-value'],[Chisquares_results[0],Chisquares_results[1]]]
Chi_square





































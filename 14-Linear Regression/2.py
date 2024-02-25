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
#
#
c1= claimants.drop('CASENUM',axis=1)
c1.head(11)
c1.describe()
#Let us check whether there are null value
c1.isna().sum()
#Ther  are serval null values 
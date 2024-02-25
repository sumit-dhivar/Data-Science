# -*- coding: utf-8 -*-
"""
Created on Tue Apr  12 15:36:26 2023

@author: sumit
"""

#A SERIES IS USED TO MODEL ONE DIMENSIONAL DATA, 
#SIMILAR TO A LIST IN Python. 
#The Series object also has a few more bits
# of data, including an #index and a name.

import pandas as pd
songs2 = pd.Series([145, 142, 38, 13],name='counts')
#It is easy to inspect the index of a series (or data frame)
songs2.index
#The index can be string based as well, 
#in which case pandas indicates
#that the datatype for the index is object (not string):

songs3 = pd.Series([145, 142, 38, 13],name='counts',
   index=['Paul', 'John', 'George', 'Ringo'])
songs3.index
songs3
#The NaN value
#This value stands for Not A Number, 
#and is usually ignored in arithmetic
#operations. (Similar to NULL in SQL).
#If you load data from a CSV file, 
#an empty value for an otherwise

#numeric column will become NaN.
import pandas as pd
f1=pd.read_csv('c:/10-python/age.csv')
f1
#None, NaN, nan, and null are synonyms
#The Series object behaves similarly to 
#a NumPy array.
import numpy as np
numpy_ser = np.array([145, 142, 38, 13])
songs3[1]
#142
#They both have methods in common
songs3.mean()
##################################
#THE PANDAS SERIES DATA STRUCTURE PROVIDES
# SUPPORT FOR THE BASIC CRUD
#operations—create, read, update, and delete.
#Creation
george= pd.Series([10, 7, 1, 22],
index=['1968', '1969', '1970', '1970'],
name='George Songs')
george
#The previous example illustrates an 
#interesting feature of pandas—the
#index values are strings and they 
#are not unique. This can cause some
#confusion, but can also be useful 
#when duplicate index items are needed.
##################################
#Reading
#To read or select the data from a series
george['1968']

george['1970']
#We can iterate over data in a series 
#as well. When iterating over a series
for item in george:
     print(item)
##############################
#Updating
#Updating values in a series can be a 
#little tricky as well. 
#To update a value
#for a given index label, 
#the standard index assignment operation 
#works
george['1969'] = 68
george['1969']

#Deletion
#The del statement appears to have 
#problems with duplicate index
s = pd.Series([2, 3, 4], index=[1, 2, 3])
del s[1]
s
#####################################
#Convert Types
#string  use.astype(str)
#numeric use pd.to_numeric
#integer use .astype(int), 
#note that this will fail with NaN
#datetime use pd.to_datetime

songs_66 = pd.Series([3, None , 11, 9],
index=['George', 'Ringo', 'John', 'Paul'],
name='Counts')

pd.to_numeric(songs_66.apply(str))
#There will be error
pd.to_numeric(songs_66.astype(str), errors='coerce')
#If we pass errors='coerce', 
#we can see that it supports many formats

#Dealing with None
#The .fillna method will replace them with a given value, -1
songs_66.fillna(-1)

#NaN values can be dropped from 
#the series using .dropna
songs_66.dropna()
###################################
#Append, combining, and joining two series
songs_69 = pd.Series([7, 16 , 21, 39],
index=['Ram', 'Sham', 'Ghansham', 'Krishna'],
name='Counts')
#To concatenate two series together, simply use the .append method.
songs=songs_66.append(songs_69)
###################################
#plotting two series
import matplotlib.pyplot as plt
fig = plt.figure()
songs_69.plot()
plt.legend()
###################################
fig = plt.figure()
songs_69.plot(kind='bar')
songs_66.plot(kind='bar', color='b', alpha=.5)
plt.legend()
#######################
data = pd.Series(np.random.randn(500),
name='500 random')
fig = plt.figure()
ax = fig.add_subplot(111)
data.hist()
##################

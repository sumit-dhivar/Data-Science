# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 09:17:44 2023

@author: sumit
"""

############ -------------NLP Pipeline------------#####################3

###Bag of Words 
#This BoW converts unstructured data to structured form 
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer 
corpus = ['At least seven Indian pharma companies are working to develop vaccine against the corona virus.', 'The deadly virus that has already infected more than 14 million globally','Bharat Biotech is the among the domestic pharma firm working on the corona virus vaccine in India']
bag_of_words_model = CountVectorizer()
print(bag_of_words_model.fit_transform(corpus).todense())
bag_of_words_df = pd.DataFrame(bag_of_words_model.fit_transform(corpus).todense())
#This will create the dataframe 
bag_of_words_df.columns=sorted(bag_of_words_model.vocabulary_)
bag_of_words_df.head()
########################################################## 
#bag of words model small 
bag_of_words_model_small = CountVectorizer(max_features=5)
bag_of_words_df_small=pd.DataFrame(bag_of_words_model_small.fit_transform(corpus).todense())
bag_of_words_df_small.columns=sorted(bag_of_words_model_small.vocabulary_)
bag_of_words_df_small.head()
import pandas as pd
import numpy as np
#read the csv
df = pd.read_csv("C:/Data_Set/spam.csv")
#check first 10 records
df.head()
#Total number of spam and ham(legitimate mails)
df.Category.value_counts()
#create one more column comprises 0 and 1
#name of columns is spam
df['Spam'] = df['Category'].apply(lambda x:1 if x=='spam' else 0)
df.shape
#-------------
#train test split
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(df.Message , df.Spam  ,test_size = 0.2)
#let us check the shape of X train and X_test data
X_train.shape
X_test.shape
#let us check the shape of Xtrain and X_test data
X_train.shape
X_test.shape
#let us check the type of X_train and y_train
type(X_train)
type(y_train) 
#################################################3 
#--------------------
#create bag of words representation using CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
v = CountVectorizer()
X_train_cv = v.fit_transform(X_train.values)
X_train_cv
#After createion of BoW , let us check the shape
X_train_cv.shape
#------------------
#Train the naive bayes model 
from sklearn.naive_bayes import MultinomialNB
#Initalize the model
model = MultinomialNB()
#Train the model
model.fit(X_train_cv , y_train)
#-----------------
#create bag of words representation using CountVectorizer of X_test
X_test_cv = v.transform(X_test)
#----------------
#Evaluate Performance
from sklearn.metrics import classification_report
y_pred = model.predict(X_test_cv)
print(classification_report(y_test,y_pred)) 































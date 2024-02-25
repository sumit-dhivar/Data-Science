# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 22:36:46 2024

@author: sumit
"""
#1.1. Business Objective:
'''
The primary objective is to prepare a classification model using 
the Naive Bayes algorithm for the salary dataset.
'''

#1.2. Constraints:
'''
There are no specific constraints mentioned in the problem statement,
 so we can assume that the main goal is to build an accurate
 classification model without any specific limitations.
'''
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
email_data=pd.read_csv("Disaster_tweets_NB.csv",encoding="ISO-8859-1")

#droping Null values
email_data.isnull().sum()
#There are some null values in the data for keyword and location
email_data['location'].fillna(value='unknown', inplace=True)
email_data['keyword'].fillna(value='missing', inplace=True)
email_data.isnull().sum()#

#EDA
email_data.info()
email_data.describe()
#This is a very unstructuted Data set we will have to clean the data first
import re
def cleaning_text(i):
    w=[]
    i=re.sub("[^A-Za-z""]+"," ",i).lower()
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))

email_data.keyword=email_data.keyword.apply(cleaning_text)
email_data=email_data.loc[email_data.keyword!="",:]

email_data.location=email_data.location.apply(cleaning_text)
email_data=email_data.loc[email_data.keyword!="",:]

email_data.text=email_data.text.apply(cleaning_text)
email_data=email_data.loc[email_data.text!="",:]

email_data.isnull().sum()

email_data['keyword'].value_counts().head(10)

#Boxplot
sns.boxplot(data=email_data,x=email_data['keyword'].value_counts())
plt.title('Boxplot of keyword')
plt.xlabel('Keyword')
plt.ylabel('Values')
plt.show()

# Correlation analysis
correlation_matrix = email_data.corr()
print(correlation_matrix)
corr=email_data.corr()
sns.heatmap(corr)
plt.title('Correlation Heatmap')
plt.show()


from sklearn.model_selection import train_test_split
email_train,email_test=train_test_split(email_data,test_size=0.2)

def split_into_words(i):
    return [word for word in i.split(" ")]


emails_bow=CountVectorizer(analyzer=split_into_words).fit(email_data.keyword)
all_emails_matrix=emails_bow.transform(email_data.keyword)

train_emails_matrix=emails_bow.transform(email_train.keyword)
test_emails_matrix=emails_bow.transform(email_test.keyword)

tfidf_Transformer=TfidfTransformer().fit(all_emails_matrix)
train_tfidf=tfidf_Transformer.transform(train_emails_matrix)
test_tfidf=tfidf_Transformer.transform(test_emails_matrix)

test_tfidf.shape


from sklearn.naive_bayes import MultinomialNB as MB
classifer_mb=MB()
classifer_mb.fit(train_tfidf,email_train.location)

test_pred_m=classifer_mb.predict(test_tfidf)
accuracy_test_m=np.mean(test_pred_m==email_test.location)
accuracy_test_m
###############################################################
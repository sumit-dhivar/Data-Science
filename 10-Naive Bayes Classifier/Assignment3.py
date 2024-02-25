# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 23:34:03 2024

@author: sumit
"""
"""
# Data Dictionary
1.User ID:ID of an User | Not Relevant
2.Gender: Gender of an individual | Relevant
3.Age: Age of a person | Relevant
4.EstimatedSakary: Salary | Relevant
5.Purchased: Purchased Car or Not | Relevant(Label)
"""
import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
df = pd.read_csv('NB_Car_Ad.csv')
df
df.columns

#Here the User ID is irrelevant so we will discard it 
df.drop({'User ID'},inplace=True,axis=1)

from sklearn.model_selection import train_test_split 
df_train,df_test = train_test_split(df,test_size=0.2)

df_bow = CountVectorizer().fit(df.Gender)

df_matrix = df_bow.transform(df.Gender)
#for training messages 
train_df_matrix = df_bow.transform(df_train.Gender)
#for testing messages 
test_df_matrix = df_bow.transform(df_train.Gender)
#Learing Term weightaging and normaling on entire emails
tfidf_transformer = TfidfTransformer().fit(df_matrix)
#Preparing TFIDF for train mails
train_tfidf = tfidf_transformer.transform(train_df_matrix)
#preparing TFIDF for test mails 
test_tfidf = tfidf_transformer.transform(test_df_matrix)
test_tfidf.shape
####Now let us apply this to Naive Bayes 

from sklearn.naive_bayes import MultinomialNB as MB 
classifier_mb = MB()
classifier_mb.fit(train_tfidf,df_train.Gender)

#evaluation on test data 
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == df_test.Gender)
accuracy_test_m
























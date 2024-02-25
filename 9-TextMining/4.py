# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 08:51:57 2023

@author: sumit
"""
#How to use TFIDF
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
corpus = ['The mouse had a tiny little mouse' , 'the cat saw the mouse' , 'the cat catch the mouse' , 'the end of mouse story']

#step1 initialize count vector
cv = CountVectorizer()
#To count the total no.of TF
word_count_vector = cv.fit_transform(corpus)
word_count_vector.shape
#Now next step is to apply IDF
tfidf_transformer = TfidfTransformer(smooth_idf=True , use_idf=True)
tfidf_transformer.fit(word_count_vector)
#This matrix is in the raw matrix form , let us convert it in a DataFrame
df_idf = pd.DataFrame(tfidf_transformer.idf_ , index = cv.get_feature_names_out(), columns = ['idf_weights'])
#sort ascending 
df_idf.sort_values(by=['idf_weights'])
####################################################33#####
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Thor eating pizza , loki is eating pizza , Irom man ate pizza already",
    "Apple is announcing new iphones tommorow", 
    "Tesla is announcing new model-3 tommorow" ,
    "Google is announcing new pixel-6 tommorow",
    "Microsoft is announcing new surface tommorow",
    "Amazon is announcing new eco-dot tommmorow ",
    "I am eating biryani and you are eating grapes"
]

#let's create the vectorizer and fit the corpus and transform them accordingly
v = TfidfVectorizer()
v.fit(corpus)
transform_output = v.transform(corpus)
#let's print the vocabulary

print(v.vocabulary_)
#let's print the idf of each word:
all_feature_names = v.get_feature_names_out()

for word in all_feature_names:
    #let's get the index in the vocabulary
    indx = v.vocabulary_.get(word)
    #get the score
    idf_score = v.idf_[indx]
    print(f"{word} : {idf_score}")
#------------------------------------------------------------------------------
from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    "Thor eating pizza , loki is eating pizza , Irom man ate pizza already",
    "Apple is announcing new iphones tommorow", 
    "Tesla is announcing new model-3 tommorow" ,
    "Google is announcing new pixel-6 tommorow",
    "Microsoft is announcing new surface tommorow",
    "Amazon is announcing new eco-dot tommmorow ",
    "I am eating biryani and you are eating grapes"
]

#let's create the vectorizer and fit the corpus and transform them accordingly
v = TfidfVectorizer()
v.fit(corpus)
transform_output = v.transform(corpus)
#let's print the vocabulary

print(v.vocabulary_)
#let's print the idf of each word:
all_feature_names = v.get_feature_names_out()

for word in all_feature_names:
    #let's get the index in the vocabulary
    indx = v.vocabulary_.get(word)
    #get the score
    idf_score = v.idf_[indx]
    print(f"{word} : {idf_score}")

import pandas as pd
#read the data into a pandas dataframe
df = pd.read_csv("Ecommerce_data.csv")
print(df.shape)
#check the distribution of labels
df['label'].value_counts()
#Add a new column which gives a unique number to each of these labels

df['label_num'] = df['label'].map({
    'Household' : 0,
    'Books' : 1 , 
    'Electronics' : 2,
    'Clothing & Accessories' :3
    })

#checking the results
df.head(5)
from sklearn.model_selection import train_test_split
X_train ,X_test , y_train , y_test = train_test_split(
    df.Text,
    df.label_num, 
    test_size=0.2,#20% samples will go to test dataset
    random_state=2022,
    stratify= df.label_num
)

print("Shape of X_train : " , X_train.shape)
print("shape of X_test : " , X_test.shape)
y_train.value_counts() 
y_train.value_counts()
############################################################################
#Apply Classifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.pipeline import Pipeline 
from sklearn.metrics import classification_report

#1. Create a pipeline object 
clf = Pipeline([('vectorizer_tfidf', TfidfVectorizer()),('KNN',KNeighborsClassifier())])
#2. fit with X_train and Y_train 
clf.fit(X_train,y_train)

#3. get the predictions for X_text and store it in y_pred 
y_pred = clf.predict(X_test)
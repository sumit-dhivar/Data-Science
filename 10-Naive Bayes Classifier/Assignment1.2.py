# -*- coding: utf-8 -*-
"""
Created on Mon Jan 29 22:34:15 2024

@author: sumit
"""

""" Buisness Objective:-
1) Salary Prediction and Compensation Planning: Use features such as education,
    occupation, and work hours to predict salary levels, helping in compensation
    planning and budgeting.
2) Work-Life Balance Optimization: Examine the relationship between work hours and 
    various demographic factors to improve work-life balance and employee satisfaction.
3) Targeting for Education Services: Leverage education-related features to identify 
potential customers for educational services or products.

# Data Dictionary
1.Age: The age of the individual | Relevant | 
2.Workclass: The type of employment or work arrangement (e.g., private, self-employed, government).| Relevant | 
3.Education: The highest level of education achieved by the individual.| Relevant
4.Educationno: The numerical representation of the education level.|Irrelevant
5.Maritalstatus: The marital status of the individual (e.g., married, single, divorced).| Relevant
6.Occupation: The type of job or profession the individual is engaged in. | Relevant
7.Relationship: The person's relationship status (e.g., husband, wife, own-child, unmarried). | Relevant
8.Race: The racial background or ethnicity of the individual. | Irrelevant
9.Sex: The gender of the individual (male or female). | Relevant
10.Capitalgain: The capital gains of the individual. | Relevant
11.Capitalloss: The capital losses of the individual.| Relevant
12.Hoursperweek: The number of hours the individual works per week.| Relevant
13.Native: Native country or place of origin.| Relevant
14.Salary: The income level or salary of the individual.| IrRelevant(Likely the target variable)
"""
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
#loading data
email_data = pd.read_csv('SalaryData_Test.csv')
email_data.columns
####------------------EDA---------------------###
email_data.info()
email_data.describe()

#Histogram for education
plt.figure(figsize=(18, 6))
plt.hist(email_data['education'],kde=True)
plt.title('Histogram of keyword')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.show()

################################################################

################################################################

# Correlation analysis
correlation_matrix = email_data.corr()
print(correlation_matrix)

corr=email_data.corr()
sns.heatmap(corr)
plt.title('Correlation Heatmap')
plt.show()
##########cleaning of data 
import re 

def cleaning_text(i):
    w=[]
    i = re.sub("[^A-Za-z""]+"," ",i).lower()
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))
#### Testing above function with some test text \

email_data.maritalstatus = email_data.maritalstatus.apply(cleaning_text)
email_data = email_data.loc[email_data.maritalstatus != "",:]
from sklearn.model_selection import train_test_split 
email_train = pd.read_csv('SalaryData_Test.csv')
email_test = email_data
#creating matrix of token counts for entire text document

def split_into_words(i):
    return [word for word in i.split(" ")]

emails_bow = CountVectorizer(analyzer=split_into_words).fit(email_data.maritalstatus)
all_emails_matrix = emails_bow.transform(email_data.maritalstatus)
#for training messages 
train_emails_matrix = emails_bow.transform(email_train.maritalstatus)
#for testing messages 
test_emails_matrix = emails_bow.transform(email_test.maritalstatus)
#Learing Term weightaging and normaling on entire emails
tfidf_transformer = TfidfTransformer().fit(all_emails_matrix)
#Preparing TFIDF for train mails
train_tfidf = tfidf_transformer.transform(train_emails_matrix)
#preparing TFIDF for test mails 
test_tfidf = tfidf_transformer.transform(test_emails_matrix)
test_tfidf.shape
 
####Now let us apply this to Naive Bayes 

from sklearn.naive_bayes import MultinomialNB as MB 
classifier_mb = MB()
classifier_mb.fit(train_tfidf,email_train.Salary)

#evaluation on test data 
test_pred_m = classifier_mb.predict(test_tfidf)
accuracy_test_m = np.mean(test_pred_m == email_test.Salary)
accuracy_test_m

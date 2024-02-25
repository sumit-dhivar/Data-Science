# -*- coding: utf-8 -*-
"""
Created on Fri May 26 08:19:59 2023

@author: ASUS
"""

import nltk
nltk.download('punkt')
sentence_data = 'The first sentence is abput Python . The second is about Django. You can learn Python,Django and Data Analytics here.'
nltk_tokens = nltk.sent_tokenize(sentence_data)
print(nltk_tokens)

#==================================================
#Non - Engilish tokenization 
import nltk 
german_tokenizer = nltk.data.load('tokenizers/punkt/german.pickle')
german_tokens = german_tokenizer.tokenize('wie geht as Ihnen? Gut, danke.')
print(german_tokens)
#===================================================
#Word Tokenization
import nltk 
word_data = 'It is originate from the ides that there are readers who prefer reading.'
nltk_tokens = nltk.word_tokenize(word_data)
print(nltk_tokens)


import nltk
nltk.download('stopwords') 
#it will download a file woth english stopwords
#Verifying the stopwords 

from nltk.corpus import stopwords
stopwords.words('english')

#==============================================
from nltk.corpus import stopwords
en_stops = set(stopwords.words('english'))
all_words = ['There' , 'is', 'is', 'a', 'tree','near','the','river']
for word in all_words: 
    if word not in en_stops:
        print(word)

import nltk 
nltk.download('omw-1.4')
nltk.download('wordnet')
from nltk.corpus import wordnet 

synonyms = []
for syn in wordnet.synsets("Soil"):
    for lm in syn.lemmas():
        synonyms.append(lm.name())

print(set(synonyms))        

from nltk.corpus import wordnet 
antonyms = []
for syn in wordnet.synsets("Soil"):
    for lm in syn.lemmas():
        if lm.antonyms():
            antonyms.append(lm.antonyms()[0].name())

print(set(synonyms))    









































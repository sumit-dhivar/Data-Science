# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 08:24:27 2023

@author: sumit
"""
import re
sentence5 = "sharat twitted ,Witnessing 70th republic day India from Rajpath,\new Delhi, Mesmorizing performance by Indian Army!"
re.sub(r'([^\s\w]|_)+',' ',sentence5).split()
####Extracting n-grams 
#n-gram can be extracted using three techniques 
#1.custom defined function 
#2.NLTK 
#3.TextBlob 
###########################################33 
#Extraction n-gram using custom defined funciton 

import re 
def n_gram_extractor(input_str,n):
    tokens = re.sub(r'([^\s\w]|_)+',' ',input_str).split()
    for i in range(len(tokens)-n+1):
        print(tokens[i:i+n])
        
n_gram_extractor("The cute little girls is playing with the kitten", 2)
n_gram_extractor("The cute little girls is playing with the kitten", 3)
n_gram_extractor("The cute little girls is playing with the kitten", 6)

#######################################################################

from nltk import ngrams 
#extraction n-grams with nltk 
list(ngrams("The cute little girls is playing with the kitten".split(),2))
list(ngrams("The cute little girls is playing with the kitten".split(),3))
################################################################### 
#pip install textblob
from textblob import TextBlob 
blob=TextBlob("The cute little girls is playing with kitten.")
blob.ngrams(n=2)
blob.ngrams(n=3)
####################################################################
###Tokenization using keras 
sentence5
from keras.preprocessing.text import text_to_word_sequence
text_to_word_sequence(sentence5)
#Here we will get the list of all the words without punctuation in the lower case 
###########################################
#tokenization using textblob 
from textblob import TextBlob
blob = TextBlob(sentence5)
blob.words
#Here we will get the list of all the words without punctuation as it i.e same case
###################################################### 
#Tweet tokenizer 
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()
tweet_tokenizer.tokenize(sentence5)
#########################################################
##Multi-Word_Expression
from nltk.tokenize import MWETokenizer 
sentence5
mwe_tokenizer = MWETokenizer([('republic','day')])
mwe_tokenizer.tokenize(sentence5.split())
mwe_tokenizer.tokenize(sentence5.replace('i',' ').split())

############################################################
###Regular Expression Tokenizer 
from nltk.tokenize import RegexpTokenizer
reg_tokenizer=RegexpTokenizer('\w+|\$[\d\.]+|\S+')
#Try this on regx101 and see the explaination if you don't understand it.
reg_tokenizer.tokenize(sentence5)
############################################################
#White space tokenizer 
from nltk.tokenize import WhitespaceTokenizer
wh_tokenizer=WhitespaceTokenizer()
wh_tokenizer.tokenize(sentence5)
###############################################################
from nltk.tokenize import WordPunctTokenizer
wp_tokenize=WordPunctTokenizer()
wp_tokenize.tokenize(sentence5)
############################################################### 
sentence6 = 'I love playing cricket.Cricket players practices hard in their innings'
from nltk.stem import RegexpStemmer
regex_stemmer=RegexpStemmer('ing$')
' '.join(regex_stemmer.stem(wd) for wd in sentence6.split())
####################################################################
sentence7='Before eating ,it would be nice to sanitize your hands'
from nltk.stem.porter import PorterStemmer
ps_stemmer=PorterStemmer()
words=sentence7.split()
" ".join([ps_stemmer.stem(wd) for wd in words])
################################################################## 
###Lemmatizatin 
import nltk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
nltk.download('wordnet')
lemmatizer=WordNetLemmatizer()
sentence8="The codes executed today are for  better that what we executing generally"
words=word_tokenize(sentence8)
" ".join([lemmatizer.lemmatize(word) for word in words])
####################################################################### 
###Singularize and Pluralization 
from textblob import TextBlob
sentence9 = TextBlob("She sells seashells on the seashore")
words=sentence9.words
words
#we want to make word[2] i.e. seashells in singular form 
sentence9.words[2].singularize()
#We want word 5 i.e seashore in plural form 
sentence9.words[5].pluralize()
###########################################################################
#Language transformation from spanish to English 
from textblob import TextBlob
en_blob = TextBlob(u'muy bien')
en_blob.translate(from_lang='es', to='en')
#es:spanish en:English
##################################################################### 
##custom stopwords removal 
from nltk import word_tokenize
sentence9 = ("She sells seashells on the seashore")

custom_stop_word_list=['she','on','the','am','is']
words = word_tokenize(sentence9)
" ".join([word for word in words if word.lower()
          not in custom_stop_word_list])
###Select words which are not in defined list
#######################################################################
#extracting general features form the raw text 
#number of words  
#detect presence of wh word 
#popularity 
#subjectivity 
#language identification 
############33 
#To identify the number of words 
import pandas as pd 
df = pd.DataFrame([['This vaccine for covid-19 will be announced on 1st August '],
                   ['Do you know how much expectaion the world population is having from this research?'],
                   ['The risk of virus will come to an end on 31st July']])
df.columns =['text']
df
#Now let us measure the numbr of words 
from textblob import TextBlob 
df['number_of_words'] = df['text'].apply(lambda x:len(TextBlob(x).words))
df['number_of_words'] 
########################################################################
#Detect the presence of words wh 
wh_words=set(['why','who','which','what','where','when','how'])
df['is_wh_words_present']=df['text'].apply(lambda x:True if len(set(TextBlob(str(x)).words).intersection(wh_words))>0 else False)
df['is_wh_words_present']

################################################## 
#polarity of the sentence 
df['polarity']=df['text'].apply(lambda x:TextBlob(str(x)).sentiment.polarity)
df['polarity']
sentence10 = 'I like this example very much'
pol = TextBlob(sentence10).sentiment.polarity
pol
sentence10 = 'This was helpful example but I would prefer another one'
pol=TextBlob(sentence10).sentiment.polarity 
pol 
sentence10="This is my personal opinion that it was helpful example but I would prefer another one"
pol=TextBlob(sentence10).sentiment.polarity 
pol 

sentence10="This is not my personal opinion that it was not helpful example but I would prefer another one"
pol=TextBlob(sentence10).sentiment.polarity 
pol 

######################################################################### 
###subjectivity of the dataframe df and check whether there is personal opinion or not 
df['subjectivity'] = df['text'].apply(lambda x:TextBlob(str(x)).sentiment.subjectivity)
df['subjectivity']

##################################################### 
#To find language of the sentence, this part of codee will get http error 
df['language'] = df['text'].apply(lambda x:TextBlob(str(x)).detect_language())


















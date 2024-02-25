# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 16:25:59 2023

@author: sumit
"""

sentence = "we are learning TextMining from Sanjvani AI"
#####if we want to know position of learning 
sentence.index("learning")
#####It will shoow learing is at position 7 
###This is goino to show charecter position from 0 including 
################################################################
#we want to know position TextMining word 
sentence.split().index("TextMining")
########It will split the words in list and count the position 
####If you want to see the list select sentence.split() and  
#it will show at 3 
##################################################################
#Suppose we wannt to print any word in reverse order 
sentence.split()[2][::-1]
####[start:end end:-1(start)] will start from -1,-2,-3 till the start 
#For the word TextMining
sentence.split()[3][::-1]
# gniniMtxeT
######################################################################
#Suppose we want to print the first & last word of the sentence.
words = sentence.split()
print(words[0],words[-1]) 
#Now we want to concate the first and last letter 
new= words[0] +" "+ words[-1]
new
#########################################################################
#We want to print even words from the sentenc
even_words = [i for i in words if len(i)%2==0]
even_words
#The words of odd length will not be printed
[words[i] for i in range(len(words)) if i%2 == 0]
#The words at the even position will be printed
##############################################################################
sentence
#Now we want to display only AI 
sentence[-3:]
##IT will start from -3,-2,-1 i.e AI 
#########################################################################
#Suppose we want to display entire sentence in reverser order 
sentence[::-1]
############################################################################### 
#Suppose we want to select each word and print in reversed order 
words 
print( " ".join(words[::-1]for word in words))
#Words are getting reverser 


##############################################################################
#tokenization 
import nltk
nltk.download('punkt')
from nltk import word_tokenize 
words = word_tokenize("I am reading NLP Fundamentals")
print(words)
##########################################
#parts of speech (PoS) tagging 
nltk.download('averaged_perceptron_tagger')
nltk.pos_tag(words)
#######It os going mention parts of speech 
################################3###############33 
#stop words from NLTK library 
from nltk.corpus import stopwords 
stop_words = stopwords.words('English')
##You can verify 179 stop words in variable explorer 
print(stop_words)
sentence1 = "I am learning NLP:It is one of the most popular library."
#First we will tokenize the sentence 
sentence_words = word_tokenize(sentence1) 
print(sentence_words) 
##Now let us filter the sentence1 using stop_words 
sentence_no_stops = " " . join([words for words in sentence_words if words not in stop_words])
sentence_no_stops
sentence1 
#you can notice that am, is, of, the, most, popular, in are missing from the result
###############################################################################
#suppose we want to replace words in string 
sentence2 = 'I visited My from IND on 14-02-19'
normalized_sentence = sentence2.replace("My" , "Malaysia").replace("IND", "India")
normalized_sentence = normalized_sentence.replace("-19","-2020")
print(normalized_sentence)
############################################################################### 
#suppose we want auto correction in the sentence 
from autocorrect import Speller
#declare the function Speller defined for English 
spell = Speller(lang='en')
spell("Engilish")
spell("Fonnd")
spell("Sumit") 
##################################################################
#suppose we seant to correct whole sentence 
sentence3 = "Natureal lanagage processsin dealsls wtih teh atr of extracctign ssentimenst"
##let us first tokenize this sentence 
sentence3 = word_tokenize(sentence3)
corrected_sentence = " " . join([spell(word) for word in sentence3])
print(corrected_sentence)

###############################################################################
#stemming
stemmer = nltk.stem.PorterStemmer()
stemmer.stem("Programming")
stemmer.stem("Programmed")
stemmer.stem("jumping")
stemmer.stem("jumped")

###############################################################################
#lematizer
#lematizer looks into dictionary words
#it maps into original dictionary words
nltk.download("wordnet")
from nltk.stem.wordnet import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()
lemmatizer.lemmatize("programed")
lemmatizer.lemmatize("programs")
lemmatizer.lemmatize("battling")
lemmatizer.lemmatize("amazing")

#-----------------------------------------------------------------------------------
#chunking (Shallow Parsing) Identifying named entities
nltk.download("maxent_ne_chunker")
nltk.download('words')
sentence4 = "We are learning NLP in python by SanjivaniAI based in India"
#first we will tokenize
words = word_tokenize(sentence4)
nltk.download('averaged_perceptron_tagger')
words = nltk.pos_tag(words)
i=nltk.ne_chunk(words,binary=True)
[a for a in i if len(a)==1]
###############################################################################
#sentence tokenization 
from nltk.tokenize import sent_tokenize 
sent = sent_tokenize("we are leatning NLP in Python. Delivered by SanjivaniAI. Do you know where it is located? It is in Kopargaon")
sent
#['we are leatning NLP in Python.',
 # 'Delivered by SanjivaniAI.',
 # 'Do you know where it is located?',
 # 'It is in Kopargaon']

###############################################################################
from nltk.wsd import lesk 
sentence1 = "keep your savings in the bank"
print(lesk(word_tokenize(sentence1),'bank'))
# Synset('savings_bank.n.02')
#####################################################
sentence2 = " It is so risky to drive over the banks of river."
print(lesk(word_tokenize(sentence2),'bank'))
 # Synset('bank.v.07') 
#############################
# Synset('bank.v.07')  is a slope in athe turn of a road or track; 
#the outside is higher than the inside in order to reduce the 
###########
#'bank' has multiple meanings. if you want to find exact meaning 
#execute the following code 
#the definitions for 'bank' can be seen  here:
from nltk.corpus import wordnet as wn 
for ss in wn.synsets('bank'): print(ss, ss.definition())

# Synset('bank.n.01') sloping land (especially the slope beside a body of water)
# Synset('depository_financial_institution.n.01') a financial institution that accepts deposits and channels the money into lending activities
# Synset('bank.n.03') a long ridge or pile
# Synset('bank.n.04') an arrangement of similar objects in a row or in tiers
# Synset('bank.n.05') a supply or stock held in reserve for future use (especially in emergencies)
# Synset('bank.n.06') the funds held by a gambling house or the dealer in some gambling games
# Synset('bank.n.07') a slope in the turn of a road or track; the outside is higher than the inside in order to reduce the effects of centrifugal force
# Synset('savings_bank.n.02') a container (usually with a slot in the top) for keeping money at home
# Synset('bank.n.09') a building in which the business of banking transacted
# Synset('bank.n.10') a flight maneuver; aircraft tips laterally about its longitudinal axis (especially in turning)
# Synset('bank.v.01') tip laterally
# Synset('bank.v.02') enclose with a bank
# Synset('bank.v.03') do business with a bank or keep an account at a bank
# Synset('bank.v.04') act as the banker in a game or in gambling
# Synset('bank.v.05') be in the banking business
# Synset('deposit.v.02') put into a bank account
# Synset('bank.v.07') cover with ashes so to control the rate of burning
# Synset('trust.v.01') have confidence or faith in


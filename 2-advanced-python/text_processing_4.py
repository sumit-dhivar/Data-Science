# -*- coding: utf-8 -*-
"""
Created on Tue May 16 08:23:12 2023

@author: ASUS
"""

#Tokenization 
txt = "Hello! I am coming Regualarly."
x = txt.split()
print(x)
print(type(x))

"""Special charecters"""
import re#Function to remove sp
def remove_sp_chr(txt):
    #define the pattern to keep 
    pat = r'[a-zA-z0-9.,!?:;\"\'\s]'
    return re.sub(pat,' ',txt)
remove_sp_chr('Number@ # 007 ~ Should` Be * present tommorow!')


import nltk
def get_stem(txt):
    stemmer  = nltk.porter.PorterStemmer()
    txt = "".join([stemmer.stem(word) for word in txt.split()])
    print(txt)
get_stem("Going collecting properly")

#matching text at the start or end of string
filename = "spam.ipy"
filename.endswith('.ipy')

#=============================================
area_name = '6th lane west Andheri'
area_name.endswith('west Andheri')

#=============================================
choices = ('http:','ftp:')
url = 'http://www.python.org'
url.startswith(choices)

#String slicin
g = '0123456789 10'
S = "SumitDhivar"
print(S[5:9])
print(S[-6:-1])
print(S[7:2:-1])#Printing in reverse order
S[-4:]=='ivar'

from fnmatch import fnmatch,fnmatchcase 
names = ['Dat1.csv','Dat2.csv','config.ini','foo.py']
[name for name in names if fnmatch(name,'Dat*.csv')]

names = ['Andheri East', 'Parle East', 'Dadar West']
[name for name in names if fnmatch(name,'*East')]
# ['Andheri East', 'Parle East'] 

names = ['ab adfh','ab kafghu','chu asdgf']
[name for name in names if fnmatch(name,'ab*')]


adresses = [
    '5412 N Clark ST',
    '1060 W Addison ST',
    '1039 W Granville Ave',
    '2122 N Clark ST',
    '4802 N Broadway',]
[c for c in adresses if fnmatch(c,'*ST')]

#Matching and searching for text patterns
text = 'yeah, but no, but yeah, but no,but yeah'
#Wxact Match 
txt == 'yeah'
#False 
#Match at start or end 
text.startswith('yeah')
#True 
text.endswith('yeah')
#True 
text.endswith('No')
#False 
#Search for the location if the first occurence 
text.find('no')
#10

#===============================================
text1 = '11/27/2012'
text2 = 'Nov 27, 2012'
import re  
#Simple matching : \d+ means match one or more digits 
if re.match(r'\d+/\d+/\d+',text1):
    print('yes')
else:
    print('no')
#yes
if re.match(r'\d+/\d+/\d+',text2):
    print('yes')
else:
    print('no')
#no
#==================================================
import re
d1 = '22-33-1345'
if re.match(r'\d{2}-\d{2}-\d{4}',d1):
    print('yes')
else:
    print('no')   
d1 = '22/33/1345'    
if re.match(r'\d+/\d+/\d+',d1):
    print('yes')
else:
    print('no')    
#@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@    
text = 'This is artificial intelligence era!!!'
lst = text.split()
print(lst)


import re
text = 'India: India has great history: In 2023 India is leading to its glorious future!'    
x = re.split(r'(?:,\;|\s)\s*',text)   
print(x)
    
import re
import string
def rem_functtion(text):
    text = ''.join([c for c in text if c not in string.punctution()])    
    return text
rem_functtion(text) 

import re 
pat = r'[.,!?/:;\"\'\s]'
re.sub(pat,'',text)    

t1 = "Rama went to Haridwar to get GangaJal."
t2 = "ahdf safihad dhfoia gangaJal"
t3 = "akdh saf sdnvm Gangajal."
#Check the text gangajal
ch = ('gangajal.','Gangajal.','gangaJal','GangaJal.')
t1.endswith(ch)   
t2.endswith(ch)
t1.startswith('Rama')    

A = 'I like Mango.'
print(A[-1:-8:-1])
print(A[6:-1])

S = 'I had been visited Pune on 11/5/2023'
print(S[-10:])

#Searching and Replacing Text
text = 'yeah, but no, but yeah, but no,but yeah'
text.replace('yeah','yup')    

import re 
text  = "17/5/2023"
re.sub(r'(\d+)/(\d+)/(\d+)',r'\2-\1-\3',text)    


import re 
datepat = re.compile(r'(\d+)/(\d+)/(\d+)')
datepat.sub(r'\3-\1-\2',text)    

newtext,n = datepat.subn(r'\3-\1-\2',text)    
newtext    

#Searching and replacing Case sensitive text
text = 'UPPER PYTHON, lowerr python','middle Python'

re.findall('python',text,flags = re.IGNORECASE)    
re.sub('python','snake',text,flags = re.IGNORECASE)

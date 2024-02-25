# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:00:19 2023

@author: sumit
"""

import re 
import string
import nltk
text = 'UPPER Python, lower python, Mixed Python'
def matchcase(word):
    
    def replace(m):
        text = m.group()
        if text.isupper():
            return word.upper()
        elif text.islower():
            return word.lower()
        elif text[0].isupper():
            return word.capitalize()
        else:
            return word 
    return replace 
re.sub('python',matchcase('snake'),text,flags=re.IGNORECASE)

#Lorem Ipsum is simply dummy text of the printing and typesetting industry. Lorem Ipsum has been the industry's standard dummy text ever since the 1500s, when an unknown printer took a galley of type and scrambled it to make a type specimen book. It has survived not only five centuries, but also the leap into electronic typesetting, remaining essentially unchanged. It was popularised in the 1960s with the release of Letraset sheets containing Lorem Ipsum passages, and more recently with desktop publishing software like Aldus PageMaker including versions of Lorem Ipsum.
str_pat = re.compile(r'\"(.*?)\"')
t1 = 'Computer says "no."'
str_pat.findall(t1)
#['no.']
#---------------------------------------------
t2 = 'Computer Says "no." Phone says "yes"'
str_pat.findall(t2)
#['no." Phone says "yes']
#after conerting 
#in this example the pattern re.compile(r'\"(.*)\"') to re.compile(r'\"(.*?)\"')
#the output is ['no.', 'yes']

comment = re.compile(r'/\*(.*?)\*/')
text1 = '/*This is comment */'
comment.findall(text1)

comment = re.compile(r'/\*((?:.|\n)*?)\*/')
text2 = ''''
/*This is a 
Multiline comment*/
'''
comment.findall(text2)


#Removing numbers from the text 
def remove_numbers(text):
    result = re.sub(r'\d+', '', text )
    return result

input_str = "There are 3 balls in the bag, and 12 in the car."
remove_numbers(input_str)
#-------------------------------------------------------
import re
import nltk 
import inflect
p = inflect.engine()
#convert number into words
def convery_num(text):
    #split  string into list of words
    temp_str = text.split()
    #initialize empty list 
    new_string = [] 
    for word in temp_str:
        #if word is a digit convert the number into digit 
        #to numbers and append into the new string list 
        if word.isdigit():
            temp = p.number_to_words(word)
            new_string.append(temp)
        #append the word as it is 
        else:
            new_string.append(word) 
#join the words of new_string to form a string 
    temp_str = ' '.join(new_string) 
    return temp_str
input_str = "There are 3 balls in the bag, and 12 in the car."
convery_num(input_str)

#===============================================
#Ex1:- Reverse each word of a string 
str_1 = 'My Name is Jessica'

def rev_str(txt):
    words = str_1.split() 
    
    rev_word=[c[::-1] for c in words] 
    new_sen = " ".join(rev_word)
    return new_sen
rev_str(str_1)
#====================================================
#Ex 2:  Read text file into a variable  and replace  
with open("D:/ANACONDA/Data Science/2-python/sample.txt.txt") as f1:
    print(f1.read().replace('\n',' '))

#====================================================
# You're working with Unicode strings, but need sure that
#thats all of the strings have 
# the same underlying repersentation.
s1 = 'Spicy Jalape\u00f1o'
s2 = 'Spicy Jalapen\u0303o'
print(s1)
#Spicy Jalapeño

print(s2)
#Spicy Jalapeño
s1 == s2
#False

import unicodedata
t1 = unicodedata.normalize('NFC',s1)

t2 = unicodedata.normalize('NFC',s2)
 
t1 == t2 
#True

t3 = unicodedata.normalize('NFD',s1)

t4 = unicodedata.normalize('NFD',s1)

t3 == t4

print(ascii(t3))
print(ascii(t1))

print(ascii(t2))
print(ascii(t4))
 
''.join(c for c in t1 if not unicodedata.combining(c))

#Workinng with unicode chaerecters in Regular Expression
#


import re 
num = re.compile('\d+')
#ASCII digits 

num.match('')
#<re.Match object; span=(0,3), match= '123'> 
#It's also important to be aware if soecial cases. For example, 
#matching combined with case folding: 
pat = re.compile('stra\u00dfe', re.IGNORECASE)
print(pat)
s = 'straße'

pat.match(s) #Matches 

#<re.Match object; span>






#Whitespace stripping 
s = '      Hello Sumit'
s.strip()
#Hello Sumit all the(UNNECESARY) white or blank space is removed 
s.lstrip()
#Hello Sumit
s.rstrip()
#      Hello Sumit . Here the unnecessary white space is not removed.
s = '      Hello Sumit    \n'
s.lstrip()
#'Hello Sumit    \n'
s.rstrip()
#'      Hello Sumit' 


#= ================================================
#charecter stripping 
t = '----hello========'
t.lstrip('-')
# 'hello========'
t.rstrip('=')
#'----hello'
t.strip('-=')
#---------------------------------------------------




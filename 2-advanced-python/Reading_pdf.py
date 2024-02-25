# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 08:07:59 2023
Reading Documents
@author: sumit
"""
from PyPDF2 import PdfFileReader 
# importing required module 
from PyPDF2 import PdfReader 
#creating a pdf reader object
reader = PdfReader('D:/ANACONDA/Data Science/2-python/Syllabus Sem-II.pdf')

print(len(reader.pages))
#39

#getting specific page from the pdf file 
page = reader.pages[20]

#Extracting text from page 
text = page.extract_text() 
print(text)

#=====================================================================
import re 
chat2 = 'HI: I have a problem with my older number 412888912'
pattern = 'order[^\d]*(\d)]'
matches = re.findall(pattern,chat2)
matches

























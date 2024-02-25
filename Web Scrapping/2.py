# -*- coding: utf-8 -*-
"""
Created on Sat Oct 28 08:45:26 2023

@author: sumit
"""

from bs4 import BeautifulSoup as bs
import requests 
link = "https://sanjivanicoe.org.in/index.php/contact"
page = requests.get(link)
page #<Response [200] >it means connection is succesfully established

page.content
#you will get all html source code but very crowdy text 
#let us apply html parser 
soup = bs(page.content,'html.parser')
soup

#Now the text is clean but not upto expectatioons 
#Now let us apply prettify method 
print(soup.prettify())
#the text is neat and clean 
list(soup.children)
#Finding all contents using tab 
soup.find_all('p')
#suppose you want to extract from first row 
soup.find_all('p')[1].get_text()

#Similarly from second row
soup.find_all('p')[2].get_text()

#finding text using class 
soup.find_all('div',class_='table')

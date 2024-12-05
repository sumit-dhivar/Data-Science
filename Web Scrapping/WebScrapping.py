# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 08:42:09 2024

@author: sumit
"""
import re
import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/List_of_Academy_Award%E2%80%93winning_films"
req = requests.get(url)
req

soup = BeautifulSoup(req.content)
soup.prettify()


film = list()
award = list()
year = list()
nomination = list() 
index = 0
for i in soup.findAll('td'):
    pat = re.sub("<td>(?:.*)\">|<td>|</td>|<.*>"," ", str(i))
    if(index%4 == 0):
        film.append(pat)
    elif(index%4 == 1):
        year.append(pat)
    elif(index%4 == 2):
        award.append(pat) 
    elif(index%4 == 3):
        nomination.append(pat)
    index +=1
        
    
print(award)
        
import pandas as pd
df = pd.DataFrame(data = film,columns=["Films"])        
df_award = pd.DataFrame(data = award,columns=["award"]) 
df_year = pd.DataFrame(data = year,columns=["year"]) 
df_nomination = pd.DataFrame(data = year,columns=["nomination"]) 
dataset = pd.concat([df,df_award,df_year,df_nomination],axis=1)
dataset.Films.tail(10)

for i in (film,award,year,nomination):
    print(len(i))
    
print(film[-4])

d2 = pd.DataFrame({"Films":film[:1374],'Awards':award[:1374],'Year':year[:1374],'Nomination':nomination[:]})

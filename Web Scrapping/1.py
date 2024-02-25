# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 09:12:15 2023

@author: sumit
"""

from bs4 import BeautifulSoup 
soup = BeautifulSoup(open("D:/ANACONDA/Data Science/CSV Files/sample_doc.html"),'html.parser')
#html.paser is used to arrange the HTML contents properly.
print(soup)
#it will show all the html elements extracted.
soup.text
#It will show only text 
soup.contents

#It is going to show all the html contents extracted.

soup.find('address')
soup.find_all('address')
soup.find_all('b')
soup.find_all('q')

table = soup.find('table')
table

for row in table.find_all('tr'):
    columns = row.find_all('td')
    print(columns)
    
#It will show all the rows except first row 
#Now we want to display M.Tech which is located in third row and second column 
#I need to give [3][2]
    table.find_all('tr')[3].find_all('td')[2]
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 08:40:57 2023

@author: sumit
"""

import psycopg2 as pg2

#Create a connection with pGRE
#'password is whatever password you set,  we set password in the installation
conn = pg2.connect(database='dvdrental',user='postgres',password='root',host='localhost',port=5432)

#Establisg connection and start cursor to be ready to query
cur = conn.cursor()

#Pass in a PostgreSQL query as a string
cur.execute('SELECT * FROM payment') 

#Return a tuple of the first row as Python objects
cur.fetchone()

#Return a tuples of the specified rows
cur.fetchmany(10)

#To save and index results, assign it to a variable
data = cur.fetchmany(10)

#Don't forget to close a connection
#killing the kernel or shutting down jupyter will also close it
cur.close()

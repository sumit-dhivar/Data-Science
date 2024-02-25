# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 09:21:29 2023

@author: sumit
"""

import psycopg2 as pg2

conn = pg2.connect(database='learning1',user='postgres',password='root')

cur = conn.cursor()

cur.execute("""
            Create TABLE courses(
                course_id SERIAL PRIMARY KEY,
                course_name VARCHAR(50) UNIQUE NOT NULL,
                course_instructor VARCHAR(100) NOT NULL,
                tuple VARCHAR(20) NOT NULL
                );
            """)
            
conn.commit()

cur.close()


#Adding values to DB Table using Python

import psycopg2 as pg2

conn = pg2.connect(database='learning1',user='postgres',password='root')

cur = conn.cursor()

cur.execute("INSERT INTO courses(course_name,course_instructor,tuple) VALUES('Introduction to SQL','Ram','Julia')");

cur.execute("INSERT INTO courses(course_name,course_instructor,tuple) VALUES('Analysing Survey Data in Python','Sham','Pyhton')");

cur.execute("INSERT INTO courses(course_name,course_instructor,tuple) VALUES('Introduction to ChatGPT','Ganesh','Theory')");

cur.execute("INSERT INTO courses(course_name,course_instructor,tuple) VALUES('Introduction to Statistics in R','Ramesh','R')");

cur.execute("INSERT INTO courses(course_name,course_instructor,tuple) VALUES('Introduction to SQL','Ram','Julia')");

cur.execute("INSERT INTO courses(course_name,course_instructor,tuple) VALUES('Hypothesis Testing in Python','Jayesh','Python')");


conn.commit()

cur.close()
conn.close()


conn = pg2.connect(database='learning1',user='postgres',password='root')

cur = conn.cursor() 

cur.execute('SELECT * FROM courses')

rows = cur.fetchall()

conn.commit()

for row in rows:
    print(row)
    
#---------------------------------------------------------------------------   

import psycopg2 as pg2

conn = pg2.connect(database='learning1',user='postgres',password='root')

cur = conn.cursor()

cur.execute('SELECT course_instructor, COUNT(*) FROM courses GROUP BY course_instructor')

conn.commit()
#------------------------------------------------------------------------- 
import psycopg2 as pg2

conn = pg2.connect(database='learning1',user='postgres',password='root')

cur = conn.cursor()

rows = cur.fetchall()

for row in rows:
    print(row)

cur.execute('SELECT * FROM courses ORDER BY course_instructor')

conn.commit()
rows = cur.fetchall()

for row in rows:
    print(row)

#============================================================================
import psycopg2 as pg2

conn = pg2.connect(database='learning1',user='postgres',password='root')

cur = conn.cursor()
#id,duration,fees
cur.execute('''CREATE TABLE courses_admin(
            course_id serial primary key,
            course_fees int,
            course_duration varchar(20) 
            );
''')

conn.commit()

cur.execute("INSERT INTO courses_admin(course_fees,course_duration) VALUES (3000,'20 Days')")

cur.execute("INSERT INTO courses_admin(course_fees,course_duration) VALUES (4500,'20 Days')")

conn.commit()

cur.execute("SELECT * FROM courses INNER JOIN courses_admin ON courses.course_id = courses_admin.course_id")

cur.fetchall()

cur.close()
conn.close()

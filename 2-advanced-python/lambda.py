# -*- coding: utf-8 -*-
"""
Created on Tue Apr  10 15:09:03 2023

@author: sumit
"""

def add(a,b,c):
    sum=a+b+c
    return sum
print(add(4,5,6))
add=lambda a,b,c:a+b+c
add(4,5,6)
######################
def mul(a,b,c):
    multi=a*b*c
    return multi
print(mul(6,7,8))
mul=lambda a,b,c:a*b*c
mul(6,7,8)
###########################
val=lambda *args:sum(args)
val(1,2,3,5,6)
val(1,2,3,5,7,8,9)
#########################
#*args
def myfun(*args):
    for i in args:
        print(i)
        
myfun("Hello","python","how","are","you")

myfun("Hello","python")
myfun("This","is","python")
######################
def person(name,*data):
    print(name)
    print(data)
    
person('Navin',28,'Mumbai',98576)
#############################
def person(name,**data):
    print(name)
    print(data)
    
person(name='Navin',age=28,place='Mumbai',mob_no=985060)
##############################
def person(name,**data):
    print(name)
    for i,j in data.items():
        print(i,j)
   
person('Navin',age=28,place='Mumbai',mob_no=985060)
#################################
val=lambda **data:sum(data.values())
val(a=2,b=6,c=7,d=10)
#######################
person=lambda **data:[(j) for i,j in data.items()]
person(name='Navin',age=26,place='Mumbai',mob_no=985060)
####################################
lst1=[3,4,5,6,7]
sqr=lambda lst1:[i**2 for i in lst1]
print(sqr(lst1))
#####################################

# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 16:41:03 2023
Basic of Python

@author: sumit
"""

print("Hello World")

x = 1
print(x)

print(type(x))
#converting to int

li=[1,2,3,4,5]
li.insert()

age1=input("Enter your age : ")
print(type(age1))
age2=input("Enter your age : ")
age=age1+age2
print(age)
age3=int(age)
print(age3)
print(type(age3))

a=int(input("Enter your age "))
a1=int(input("Enter your age "))
a2=a+a1
print(a2)
print(type(a2))

int_v=100
f=float(int_v)
print(f)
a='10'
f=float(a)
print(f)
b=int(a)
print(b)
print(type(b))

""" 
5 april 2023 
@author Vaibhav Bhorkade
Basic of Python
"""
#complex numbers
c1=1
c2=2j
print('c1:',c1,'c2 :',c2)
print(type(c1))
print(type(c2))
print(c1.real)
print(c2.imag)

#Boolean data type
all_ok=True
print(all_ok)
all_ok=False
print(all_ok)
print(type(all_ok))

#convert strings into bool
status=bool(input("Ok it is conform: "))
print(status)
print(type(status))

# Arithmetic Operation
home=10
away=15
print(home+away)
print(type(home+away))
print(10 * 4)
print(type(10*4))
#fractional part ignore
print(100/20)
print(type(100/20))
print(100//20)
print(type(100//20))
# Modulo operation %
print('Modulo operation 2%3 :',2%3)
print('Modulo operation 100%13 :',100%13)

#raise to
a=5
b=3
print(a ** b)
# assignment 
x=0
x+=1
#None Type
win=None
print(win is None)
print(win is not None)
print(type(win))

#indentation
num=int(input("ENter the number : "))
if num > 0:
    print(num)
    
nu=10
if nu < 0:
    print('Its negative')
else:
    print('Its not negative')

saving=float(input('Enter how much saving : '))
if saving == 0:
    print('sorry no saving')
elif saving < 500:
    print("Well done")

# iterating loop - while loop
count=0
print('starting')
while count <= 10:
    print(count)
    count+=1
# Range function
print("Print out values in a array ")
for i in range(10):
    print(i)
    print('done')

num=int(input("Enter a no check for: "))
for i in range(0,15):
    if i==num:
        break
    print(i)
    print('done')
#verticle ....    '_' is used as anonymous either i
for _ in range(0,10):
    print('.',end='')
    print()
 #horizonal .....   
for _ in range(0,10):
    print('.',end='')
    
"""
              ----Test case 1----
Write  python program to calculate BMI of a person using if elif else
5/4/23
"""
height=float(input("Please enter your height in m : "))
weight=float(input("Please enter your weight in kg : "))
BMI=round((weight/(height * height)),2)
if BMI<18.5:
    print(f"You are under weight and your BMI is {BMI}")
elif BMI>18.5 and BMI<25:
    print(f"You are normal weight and your BMI is {BMI}")
elif BMI>30 and BMI<35:
    print(f"You are obese and your BMI is {BMI}")
elif BMI>35:
    print(f"You are clinically obuse and your BMI is {BMI}")
    

"""
          ----Test case 2----
  write python code using logical operators and if elif . so as to allow 
  for roller coster also ask user age and charge ticket accordingly
  5/04/23
"""
print("Welcome to roller coster ")
height=int(input("Enter your height in cm : "))
if height>=120:
    print("You are eligible for roller coster ")
    age=int(input("Enter your age : "))
    bill=0
    if age<12:
        print("Childs ticket is $5")
        bill=5
    elif age>12 and age<18:
        print("Youths ticket is 7$")
        bill=7
    elif age>=18 and age<45:
        print("Youngs ticket is 12 $")
        bill=12
    elif age>=45 and age<55:
        print("Adults ticket is 50 $")
        bill=50
    want_photo=input("Do you want photo y/n : ")
    if want_photo=='Y':
        bill+=3
        print("Total bill is ", bill)
    else:
        print(bill)

"""
created on Thu 6/04/23
@Vaibhav Bhorkade
"""
# break loop statement
print("Only print codeif all iterations complete ")
num=int(input("Enter a number check for : "))
for i in range(0,6):
    if i==num:
        break
    print(i,' ',end='')
print("Done")

print("Only print codeif all iterations complete ")
num=int(input("Enter a number check for : "))
for i in range(0,6):
    if i==num:
        break
    print(i,' ',end='')
    print("Done")
    
# Creating Tuple
tup=(1,2,3,5,6)
print(f"tup[0]  {tup[0]}")
print(f"tup[1]  {tup[1]}")
print(f"tup[4]  {tup[4]}")
print(f"tup[3]  {tup[3]}")

#tuple can hold different type of value 
tup2=(1,"hi",True)
print(tup2)
print(type(tup2))
# iteration in tuple 
for x in tup2:
    print(x)
tup3=("apple","orange","plum","apple")
for i in range(0,4):
    print(tup3[i])
    
tup4=("apple","orange","apple","apple")
tup4.count("apple")
len(tup4)
print(tup4.index("apple"))
print(tup4.index("orange"))
tup4=("apple","orange","apple","apple")
if 'apple' in tup4:
    print("Present")
if 'banana' not in tup4:
    print("Not present")
tup5=(1,2,3,('vaibhav','gaurav'),5)
print(tup5)
print(tup5[3])

#List Operations
lst=['john','paul','george','ringo']
print(lst)
lst[1]
lst[-1]
lst[-2]
lst1=[1,2,3,4]
root_lst=['vaibhav',lst,lst1,3]
print(root_lst)

# append and extend
lst=['john','paul','george','ringo']
print(lst[1:])
print(lst[-3:-1])
lst.append('vaibhav')
print(lst)
lst.extend('gaurav')
print(lst)
lst=['john','paul','george','ringo']
#insert 
lst.insert(0,'shubham')
print(lst)


"""
 6/04/23
 Write a python program to find out is duplicate is present 
 or not
"""
lst=[1,2,3,4,5,6,7,2,9]
def duplicate(lst):
    for i in range(len(lst)-1):  #-1 because start with 0
        for j in range(1,len(lst)): # start 1 j for another travel
           if(lst[i]==lst[j]):
            return True
    return False
print(duplicate(lst))

a=[]
b=int(input("En  "))
a.append(b)
print(a)

"""
Pyramid
"""
for i in range(4):
    for j in range(4):
        print("*",end=' ')
    print()
    
for i in range(4):
    for j in range(4-i):
        print("*",end=' ')
    print()
    
for i in range(4):
    for j in range(i+1):
        print("*",end=' ')
    print()

st='abcd'
st[::-1]

# palimdrome python program
a=input("Enter the data : ")
def palimdrome(a):
    if a==a[::-1]:
        print("It is palimdrome string ")
    else:
        print("It is not palimdrome string ")
palimdrome(a)

# find minimum number in list
lst=[1,3,6,2,9,0]
a=lst[0]
for i in lst:
    if i<a:
        min=i
print(f"The minimum number in list is {min}")

# find maximum number in list

lst=[1,3,6,2,9,0]
a=lst[0]
for i in lst:
    if i>a:
        max=i
print(f"The maximum number in list is {max}")
"""
Created on Fri Apr  7 08:30:59 2023

@author: Rahul Raje
"""

# Remove Method
a=['vaibhav','gaurav','ketan','sachin']
print(a)
a.remove('ketan')
a.pop(1)
print(a)    
# insert
a=['vaibhav','gaurav','ketan','sachin']
a.insert(1,'Shubham')
print(a)
#list concantation
l1=[1,2,3,4]
l2=[5,6,7,8]
l3=l1+l2
print(l3)

# creating set
basket={'banana','apple','orange','pear','apple'}
print(basket)

for item in basket:
    print(item)
# insert into set
basket.update(['hello','hi'])
print(basket)

basket.add("mango")
print(basket)
# min and max 
print(len(basket))
basket2={1,5,77,2,3}
print(max(basket2)) 
print(min(basket2)) 

# Remove method
print(basket.remove('apple'))  

print(basket)    

print(basket.discard('hello'))
print(basket)

#Set OPeration
s1={'apple','orange','banana'}
s2={'grape','lime','banana'}
print("union :",s1|s2)
print("Integratio :",s1&s2)
print("diiff :",s1-s2)
print("diiff :",s2-s1)

#dictonories
dic={
      'maharashtra':'Mumbai',
      'Gujrat':'Ahmadabad',
      'Up':'lakhnow'}
print(dic)
print('dict[Maharashtra]:',dic['maharashtra'])
dic.pop('Up')
print(dic)
# adding new element
dic['west']='kol'
print(dic)
# deleting new element
del dic['west']
print(dic)

# interartion
for states in dic:
    print(states,end=', ')

#values
print(dic.values())
print(dic.keys())
print(dic.items())


"""
Created on Mon Apr 10 08:20:02 2023

@author: Vaibhav Bhorkade
"""
 
     
dic={
          'maharashtra':'Mumbai',
          'Gujrat':'Ahmadabad',
          'Up':'lakhnow'
          } 
print(dic)
print('mumbai' in dic )
print('Up' in dic )

print(len(dic))   
# dictionary can have tuple value
seasons={'summer':('feb','march','april','may'),
         'rainy':('june','july','august','sept'),
         'winter':('oct','nov','dec','jan')}
print(seasons['rainy'])
print(seasons['rainy'][1])

seasons={'summer':['feb','march','april','may'],
         'rainy':['june','july','august','sept'],
         'winter':['oct','nov','dec','jan']}
print(seasons['rainy'])
print(seasons['rainy'][1])

dict2={
       'brand':'Maruti',
       'model':'breeza',
       'year':2022,
       'year':2021}
print(dict2)
for x in dict2:
    print(x)
for x in dict2:
    print(dict2[x])
    
#values accessed
for x in dict2.values():
    print(x)
    
for x in dict2.keys():
    print(x)

mydict=dict2.copy()
print(mydict)

# function
def prime_num(num):
    for i in range(2,num):
        if(num%i==0):
            return 'It is not prime number'
            break
    return 'It is prime number'
print(prime_num(11))

#display simplay greeting
def greet(username):
    print(f"hello , {username}")
greet('Sanjivani')
# argument and parameter
def describe(animal_type,pet_name):
    print(f"I have a {animal_type}")
    print(f"My {animal_type} name is {pet_name}")
describe('dog','moti')
describe('cat','chiv')
# default value argument
def describe(animal_type,pet_name='moti'):
    print(f"I have a {animal_type}")
    print(f"My {animal_type} name is {pet_name}")
describe(animal_type='cat')

def describe(animal_type='dog',pet_name='moti'):
    print(f"I have a {animal_type}")
    print(f"My {animal_type} name is {pet_name}")
describe()

# anagram program
def are_anagram(str1,str2):
    a=list(str1.replace(""," ").lower())
    b=list(str2.replace(""," ").lower())
    print(a)
    print(b)
    if(len(a)!=len(b)):
        return False
    else:
        return(sorted(a)==sorted(b))
    
print(are_anagram("elbow","below"))
print(are_anagram("race","care"))

# sum of list which divisible 5 and 7
lst=[5,7]
lst1=[1,2,5,7,14,55,45,21,49,8,9]

def return_sum(lst):
    sum=0
    for i in range (len(lst)):
        if(lst[i]%5==0 or lst[i]%7==0):
            sum=sum+lst[i]
    return sum
print(return_sum(lst))
print(return_sum(lst1))


#write a function to reverse the sentence
def reverse_word(input):
    if input=="":
        return "You entered wrong input"
    else:
        words=input.split()
        reverse_sen="  ".join(reversed(words))
    return reverse_sen
print(reverse_word("This is india"))
print(reverse_word("I am vaibhav"))


"""
Created on Tue Apr 11 08:15:51 2023

@author: Vaibhav Bhorkade
"""
def get_formated(first_name,last_name):
    full_name=f"{first_name} {last_name}"
    return full_name
cricketer=get_formated('Rohit','sharma')
print(cricketer)

# dictionary function even return
def build_person(first_name,last_name):
    person={'first':first_name,'last':last_name}
    return person
musician=build_person('Ram', 'Sarkar')
print(musician) 

# list to function
def greet_user(names):
    for name in names:
        msg=f"Hello {name}"
        print(msg)
usernames= ['Vaibhav' , 'Saurabh' ,'Surabhi']
greet_user(usernames)

# list to function
def greet_user(names):
    for name in names:
        msg=f"Hello {name.title()}"
        print(msg)
usernames= ['Vaibhav' , 'saurabh' ,'Surabhi']
greet_user(usernames)

# * for multiple parameter
def make_pizza(*toppings):
    print('making pizza of following toppings')
    for topping in toppings:
        print(f"- {topping}")
    #print(toppings)
make_pizza('pepperoni')
make_pizza('mushrooms','green','cheese')

#Mixing the positional and arbitaral arguments
def make_pizza(size,*toppings):
    print(f'making a {size} pizza of following toppings')
    for topping in toppings:
        print(f"- {topping}")
    #print(toppings)
make_pizza(16,'pepperoni')
make_pizza(12,'mushrooms','green','cheese')


import pizza as p
p.make_pizz(12,'mushrooms','green','cheese')

from pizza import make_pizz
make_pizz(10,'mushrooms','green','cheese')


"""
Assignment
Write a code for leaf year
"""
def leaf_year(year):
    if (((year>0) and (year%4==0)) or ((year%100==0)) and (year%400==0)):
        return True
    return False

print(leaf_year(1601))
print(leaf_year(2016))
print(leaf_year(2000))
# using else condition
def leaf_year(year):
    if ((year>0) and (year%4==0)):
        return True
    elif((year%100==0) and (year%400==0)):
     return True
    else:
        return False
print(leaf_year(2016))
print(leaf_year(2000))
print(leaf_year(1601))

"""
Assignment
Generate and display password containing 7 and 10 chracters
"""
from random import randint
SHORTEST=7
LONGEST=10
MIN_ASCII=33
MAX_ASCII=126
print(SHORTEST)

# constant in capital to declare
# camelNameing randomPassword
def randomPassword():
    randomLength=randint(SHORTEST, LONGEST)
    result=" "
    for i in range(randomLength):
        randomChar=chr(randint(MIN_ASCII,MAX_ASCII))
        #the char fun takes ASCII code as its parameter ,It returns a string containing the
        #character with that ASCII code as its result
        result=result+randomChar
    return result
print("Your random password is :",randomPassword())
print("Your random password is :",randomPassword())

#Write Python code to for fibbonacy series

def fibbonac(n):
    lst=[0,]
    previous=0
    current=1
    lst.append(current)
    for i in range(n-1):
        previous,current=current,previous+current
        lst.append(current)
    return lst
print(fibbonac(10))
  
# import pizza 
from pizza import make_pizz as mp
mp.make_pizz(16,'pepperoni')

"""
Created on Wed Apr 12 08:22:15 2023

@author: Vaibhav Bhorkade
"""
#Using '  as' to give a functions an Alias
from pizza import make_pizza as mp
mp(16,'pepperoni')

import pizza as p
p.make_pizza(16,'pepperoni')

from pizza import *
make_pizza(12, 'potato')


#Scope of Variable 
x = x+1  
x = 6 
print(x)

#You cannot reference a variable until it has been assigned a value 

#Lambda function 

# A lambda function is a small anonymous function.

# A lambda function can take any number of arguments, but can only have one expression.

x = lambda a : a + 10
print(x(5))


#Lambda functions can take any number of arguments:
    
x = lambda a, b : a * b
print(x(5, 6))

x = lambda a, b, c : a + b + c
print(x(5, 6, 2))

#The power of lambda is better shown when you use them as an anonymous function inside another function
#Use that function definition to make a function that always doubles the number you send in
def myfunc(n):
  return lambda a : a * n

mydoubler = myfunc(2)

print(mydoubler(11))
"""
File handling 
"""
with open ('handle.txt') as file_object:
    context=file_object.read()
print(context)



with open ('handle.txt') as file_object:
    context=file_object.read()
    #rstrip() remove gap or white space 
print(context.rstrip())



with open ('c:/1-python/handle.txt') as file_object:
    context=file_object.read()
    #rstrip() remove gap or white space 
print(context.rstrip())

filepath='c:/1-python/handle.txt'
with open (filepath) as file_object:
    context=file_object.read()
    #rstrip() remove gap or white space 
print(context.rstrip())

with open (filepath) as file_object:
    for line in file_object:
        print(line)

with open (filepath) as file_object:
    for line in file_object:
        print(line.rstrip())
        
with open (filepath) as file_object:
    lines=file_object.readlines()
for line in lines:
    print(line)

"""
evening session 12/4/23
Write a python code for Lottery random number
"""
from random import randrange
MIN_NUM=1
MAX_NUM=49
NUM_NUMS=6
ticket_nums=[]
for i in range(NUM_NUMS):
    rand=randrange(MIN_NUM,MAX_NUM+1)
    # print("Randf:-",rand)
    #This while loop is to check the element is present in the list or not
    while rand in ticket_nums:
        rand=randrange(MIN_NUM,MAX_NUM+1)
        # print("Randw:-",rand)
    ticket_nums.append(rand)
ticket_nums.sort()
for n in ticket_nums:
    print(n,end=" ")

"""
remove outliers
#higher element deleted
"""
value=[89,105,7,4,12,23]
retval=sorted(value)
def removeOutliers(data,num_outliers):
    retval=sorted(data)
    for i in range (num_outliers):
        retval.pop(-1)
    return retval
removeOutliers(value, 2)
    
"""
Write python code that determine whether or not password 
good or not . We define good password if it follows following
conditions:
    1.at least 8 character
    2.atleast one upper case
    3.one lower case letter
"""
def checkPassword(password):
    has_upper=False
    has_lower=False
    has_num=False
    
    for i in password:
        if (i>='A' and i<='Z'):
            has_upper=True
        elif (i>='a' and i<='z'):
            has_lower=True
        elif (len(password)>=8):
            has_num=True
    
    if (has_upper==True and has_lower==True and has_num==True):
        print("Good Password")
    else:
        print("Bad Password ")
        
a=(input("Enter a password : "))
checkPassword(a)

"""
Created on Thu Apr 13 08:14:07 2023

@author: Vaibhav Bhorkade

"""
# file handling
# if we use absult path no need to change directory
a=open('handle.txt')
a.readlines()
 
filename='handle.txt'
with open(filename) as file_object:
    lines=file_object.readlines()
    pi_string=' '
    for line in lines:
        pi_string += " "+line.rstrip()
    print(pi_string)
    print(len(pi_string))
    
    
filename='handle.txt'
with open(filename,'w') as file_object:
    file_object.write("I am a programmer")
    file_object.write("\nHello")
   
    
    
a=open('handle.txt')
a.readlines()
# new line     
filename='han.txt'
with open(filename,'w') as file_object:
    file_object.write("I am a programmer")
    file_object.write("\nHello")

# create write new file
filename='programming.txt'
with open(filename,'w') as file_object:
    file_object.write("I am a programmer")
    file_object.write("\n I love Programming")


filename='programming.txt'
with open(filename,'a') as file_object:
    file_object.write("I am a Vaibhav")
    file_object.write("\n I am computer engineer")
a=open('handle.txt')
a.read()

# appending to a file
filename='programming.txt'
with open(filename,'a') as file_object:
    file_object.write("Hello p")
    file_object.write("\n Sanjivani")
"""
Exception Handling
"""
try:
    print(5/0)
except ZeroDivisionError:
    print("You can not divide by zero")

try:
 print("GIve me two numbers , and i will divide them")
 print("Enter 'q' to Quit")
 while True:
     first_number=input("Enter the first number : ")
     if first_number=='q':
         break
     second_number=input("Enter second number : ")
     if second_number=='q':
         break
     answer=int(first_number)/ int(second_number)
     print(answer)
except ZeroDivisionError:
    print("You can not divide by zero")
    
filename='alice.txt'
try:
    with open(filename,encoding='utf-8') as f:
        contents=f.read()
except FileNotFoundError:
    print("Enter corrent file ")
    
     
    

"""
Created on Mon Apr 17 08:12:28 2023

@author: Vaibhav Bhorkade
Advanced Python
"""
# JSON - JavaScript Object NOtation  
import json
numbers=[2,3,4,5,7,11,13]
filename='numbers.json'
with open(filename, 'w') as f:
    json.dump(numbers ,f)

a=open('numbers.json')
a.read()

#Saving data json is useful
username=input("What is your name : ")
filename='username.json'
with open(filename,'w') as f:
    json.dump(username, f)
print(f"We'll remenber you mean when you back {username}")

import json
filename='username.json'
with open(filename) as f:
    username=json.load(f)
print(f"Welcome back , {username} !")

filename='username.json'
try:
    with open(filename) as f:
        username=json.load(f)
except FileNotFoundError:
    username=input("What is your name ?")
    with open(filename,'w') as f:
        json.dump(username,f)
        print(f"We will Remember you when you come back, {username}")
else:
    print(f"Welcome back ,{username}")

# list
lst=[]
for num in range(0,20):
    lst.append(num)
print(lst)

# list comprehention
lst=[num for num in range (0,20)]
print(lst)

names=['dada','kaka','mama']
lst=[name.capitalize() for name in names]
print(lst)

names=['dada','kaka','mama']
lst=[name.upper() for name in names]
print(lst)
lst=[name.lower() for name in names]
print(lst)

# List comprehension with if statement
def is_even(num):
    return num%2==0
lst=[num for num in range (10) if is_even(num)]
print(lst)



"""
Created on Tue Apr 18 08:18:40 2023

@author: Vaibhav Bhorkade
"""
###########################

lst=[f"{x}{y}"
     for x in range(4)
     for y in range(4)
     ]
print(lst)
print(len(lst))
########--3 digit--########
lst=[f"{x}{y}{z}"
     for x in range(4)
     for y in range(4)
     for z in range(4)
     ]
print(lst)
print(len(lst))
####--SET Compression --####
set1={
      x
      for x in range(3)
      }
print(set1)
###########################
#Dictionary Comprehension
dict={
      x:x*x
      for x in range(3)
      }

print(dict)
####----Generator----#####
gen=(
    x
    for x in range(3)
    )
print(gen)

for num in gen:
    print(num)
#####---accesss using next---##
gen=(x for x in range(3))
next(gen)
next(gen)
next(gen)

gen=(
    x
    for x in range(3)
    )
next(gen)
#####----Function return multiple value---##
def range_even(end):
    for num in range(0,end,2):
        yield num
for num in range_even(6):
    print(num)
######---insted of  loop we can write---#####
gen=range_even(6)
next(gen)
next(gen)

#Chaining Generators
#This  will return the length of the elements in the list
def lengths(itr):
    for ele in itr:
        yield len(ele)
#This will return the string of * (Here the itr probably have and int value so that it can multiply)

def hide(itr):
    for ele in itr:
        yield ele*"*"
passwords=["Vaibhav","Hello"]

for password in hide(lengths(passwords)):
    print(password)

####----Printing list with index -----#####
lst=["milk","egg","bread"]
for index in range(len(lst)):
    print(f" {index+1} {lst[index]}")
    
"""
Assignment
password picker
"""
import string
#pick the adjective
adjective=['sleepy','slow','smelly','wet','fat','red','orange','yellow','green','blue','purple','fluffy','white','proud','brave']
#pick the nouns
nouns=['apple','dinisaur','ball','toaster','goat','dragon','hammer','duck','panda']
#pick the words
import random
adjective=random.choice(adjective)
noun=random.choice(nouns)
number=random.randrange(0,100)
#select special char
special_char=random.choice(string.punctuation)
#create the new secure password
password = adjective + noun + str(number) + special_char
print("Your password is : %s"%password)

# while loop to generate more password
print("Welcome to password picker ")
while True:
    adjective=random.choice(adjective)
    noun=random.choice(nouns)
    number=random.randrange(0,100)
    special_char=random.choice(string.punctuation)
    password = adjective + noun + str(number) + special_char
    print("Your password is : %s"%password)
    response=input("Would you like another password (Y or n) ")
    if  response=='n':
        break

"""
ASSIGNMENT 2
"""
print("Append in list : ")
adjective=[]
while True:
    a=input("Enter the adjective : ")
    adjective.append(a)
    print(adjective)
    response=input("Would you like another adjective (Y or n) ")
    if  response=='n':
        break
nouns=[]
while True:
    b=input("Enter the noun : ")
    nouns.append(b)
    print(nouns)
    response=input("Would you like another noun (Y or n) ")
    if  response=='n':
        break
import random
import string

print("Welcome to password picker ")
while True:
    adjective=random.choice(adjective)
    noun=random.choice(nouns)
    number=random.randrange(0,100)
    special_char=random.choice(string.punctuation)
    password = adjective + noun + str(number) + special_char
    print("Your password is : %s"%password)
    #checkPassword(password)
    response=input("Would you like another password (Y or n) ")
    if  response=='n':
        break
def checkPassword(password):
    has_upper=False
    has_lower=False
    has_num=False
    
    for i in password:
        if (i>='A' and i<='Z'):
            has_upper=True
        elif (i>='a' and i<='z'):
            has_lower=True
        elif (len(password)>=8):
            has_num=True
    
    if (has_upper==True and has_lower==True and has_num==True):
        print("Good Password")
    else:
        print("Bad Password ")
        
checkPassword(password)

"""
Created on Wed Apr 19 08:26:08 2023

@author: Vaibhav Bhorkade
"""
lst=["milk","egg","bread"] 
for index in range(len(lst)):
    print(f" {index+1} {lst[index]}")
##---- same code using enumerate ---=
for index, item in enumerate(lst,start=1):
    print(f"{index} {item}")
#---- zip function ---
name=['dada','mama','kaka']
info=[9870,5040,3456]
for nm,inf in zip (name,info):
    print(nm,inf)
# ---- drawback of zip ---- mismatch item not display

name=['dada','mama','kaka','baba']
info=[9870,5040,3456]
for nm,inf in zip (name,info):
    print(nm,inf)
# zip_longest 
from itertools import zip_longest
name=['dada','mama','kaka','baba']
info=[9870,5040,3456]
for nm,inf in zip_longest (name,info):
    print(nm,inf)
# use fill value
from itertools import zip_longest
name=['dada','mama','kaka','baba']
info=[9870,5040,3456]
for nm,inf in zip_longest(name,info,fillvalue=0):
    print(nm,inf)

# use all function

#count():- use in deep learniing
from itertools import count
counter=count()
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))

#let us start from the 1 instead of 0
from itertools import count
counter=count(start=1)
print(next(counter))
print(next(counter))
print(next(counter))
print(next(counter))
###################################
#cycle()
#suppose you have repeated tasks to be done in a constant time  then we use cycle() iterator
import itertools
instructions=("eat","code","sleep")
for instruction in itertools.cycle(instructions):
    print(instruction)
#we can get the output in the infinite loop
################################

#repeat()
from itertools import repeat
x="keep patience"
for msg in repeat(x,times=3):
    print(msg)
#########################################
#group-combinations()
from itertools import combinations
player={'john','jain','sumit'}
for group in combinations(player,2):
    print(group)
#group are the variable not the special key word
from itertools import combinations
player={'john','jain','sumit'}
for x in combinations(player,2):
    print(x)
#################################


"""------------------- assigement day 19 4 23-------------------------"""
#use the admine user 
users=["admine","employee","manager","worker","staff"]
for user in users:
    if user=="admin":
        print("hello admin ,would you like to see the status report")
    elif user=="employee":
        print("hello employe")
    elif user=="manager":
        print("hello manager")
    elif user=="worker":
        print("hello worker")
    else:
        print("hello staff")

##  check the new user and old user
current_user=["ali","ahemd","fahad","aun","rana"]
new_users=["ali","rana","bilai","huzi","dula"]
for new_user in new_users:
    if new_user in current_user:
        print("person will need to enter  a new uersname")
    else:
        print("saying that the username is avilable")















# user 
users=['admin','employee','manager','worker','staff']
for user in users:
    if user=="admin":
        print("Hello admin , would you like see status report ")
    elif user=="employee":
        print("Hello employee")
    elif user=="manager":
        print("Hello manager")
    elif user=="worker":
        print("Hello worker")
    else:
        print("Hello")
    
#####################################
current_users=['Ram','shyam','vaibhav']
new_users=['gaurav','Ram','ketan']
for new_user in new_users:
    if new_user in current_users:
        print("Person will need to enter new username ")
    else:
        print("Welcome ")

import hashlib
hashlib.sha256("Vaibhav@15916".encode('utf-8')).hexdigest()
'afc4703fe925d6e2251460a668ee20bf447b62e96b43c4d1483eb2a149b84590'
len(hashlib.sha256("Vaibhav@15916".encode('utf-8')).digest())


def checkPassword(password):
    has_upper=False
    has_lower=False
    has_num=False
    
    for i in password:
        if (i>='A' and i<='Z'):
            has_upper=True
        elif (i>='a' and i<='z'):
            has_lower=True
        elif (len(password)>=8):
            has_num=True
    
    if (has_upper==True and has_lower==True and has_num==True):
        print("It is a Good Password")
    else:
        print("It is a Bad Password ")

#==============================================================================
import json
import hashlib
from itertools import count

counter = count(start=1)

while True:
    choice = int(input("1) Login\n2) Sign up\n3) Exit\nEnter choice: "))
    
    if choice == 1:
        # Get username and password from user
        username = input("Enter Username: ")
        password = input("Enter Password: ")
    
        # Hash password
        hash_pass = hashlib.sha256(password.encode("utf-8")).hexdigest()
    
        try:
            # Load existing usernames and passwords from file
            with open('usersnpass.json', 'r') as f:
                user_dict = json.load(f)
        except json.JSONDecodeError:
            user_dict = {}
    
        # Check if username exists and password matches
        if username in user_dict and user_dict[username] == hash_pass:
            print("Login successful!")
            counter = count(start=1)
        else:
            # Increment the counter
            attempts = next(counter)
            if attempts >= 4:
                print("Too many attempts. Try again later.")
                break
            else:
                print("Invalid username or password.")
                
    elif choice == 2:
        # Get username and password from user
        username = input("Enter Username: ")
        password = input("Enter Password: ")
        if checkPassword(password):
        
            # Hash password
            hash_pass = hashlib.sha256(password.encode("utf-8")).hexdigest()
        
            try:
                # Load existing usernames and passwords from file
                with open('usersnpass.json', 'r') as f:
                    user_dict = json.load(f)
            except json.JSONDecodeError:
                user_dict = {}
        
            # Check if username already exists
            if username in user_dict:
                print("Username already exists. Please choose a different username.")
            else:
                # Add new username and hashed password to dictionary
                user_dict[username] = hash_pass
                with open('usersnpass.json', 'w') as f:
                    json.dump(user_dict, f)
                print("Signup successful!")
    
    elif choice == 3:
        break
    else:
        print("Invalid choice.")
        



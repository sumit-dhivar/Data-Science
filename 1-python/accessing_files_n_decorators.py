# -*- coding: utf-8 -*-
"""
Created on Tue Apr  11 15:03:19 2023

@author: sumit
"""
import pandas as pd
f1=pd.read_csv('C:/10-python/buzzers.csv')
###############################
import os
with open('buzzers.csv') as raw_data:
     print(raw_data.read())
########################
#Reading CSV Data As Lists
import csv
with open('buzzers.csv') as raw_data:
    for line in csv.reader(raw_data):
        print(line)
################################
#Reading CSV Data As Dictionaries
import csv
with open('buzzers.csv') as raw_data:
    for line in csv.DictReader(raw_data):
        print(line)
####################################
with open('buzzers.csv') as data:
    #ignore=data.readline()
    flights={}
    for line in data:
        k,v=line.split(',')
        flights[k]=v
flights
################################
#stripping Then Splitting , Your Raw Data
with open('buzzers.csv') as data:
    ignore=data.readline()
    flights={}
    for line in data:
        k,v=line.strip().split(',')
        flights[k]=v
flights
###############################
#pre-requisite to decorators
def plus_one(number):
    number1= number + 1
    return number1
plus_one(5)

###################################
#Defining Functions Inside other Functions
def plus_one(number):
    
    def add_one(number):
        number1=number+1
        return number1

    result = add_one(number)
    return result
plus_one(4)
####################################
#Passing Functions as Arguments 
#to other Functions
def plus_one(number):
     result1= number + 1
     return result1

def function_call(function):
    result=function(5)
    return result

function_call(plus_one)
#################################
#Functions Returning other Functions
def hello_function():
    def say_hi():
        return "Hi"
    return say_hi
hello=hello_function()
hello()
#Always remember when you call hello_function()
#directly then it will display object not hi
#Therefore you need to assign it to hello first
#then call hell() function
#######################
#A Python decorator is a function 
#that takes in a function and 
#returns it by adding some functionality.
def say_hi():
    return 'hello there'

def uppercase_decorator(function):
    def wrapper():
        func = function()

        make_uppercase = func.upper()
        return make_uppercase

    return wrapper 

decorate = uppercase_decorator(say_hi)
decorate()
#However, Python provides a much easier way
# for us to apply decorators. 
#We simply use the @ symbol before 
#the function we'd like to decorate.
#####################################
@uppercase_decorator
def say_hi():
    return 'hello there'

say_hi()
#################################
#Applying Multiple Decorators 
#to a Single Function
#We can use multiple decorators 
#to a single function. However, 
#the decorators will be applied in the order
# that we've called them.
def split_string(function):
    def wrapper():
        func = function()
        splitted_string = func.split()
        return splitted_string

    return wrapper
@split_string
@uppercase_decorator
def say_hi():
    return 'hello there'
say_hi()
###########################################
##########################################
#Nested Function
def outer(x):
    def inner(y):
        return x + y
    return inner

add_five = outer(5)
result = add_five(6)
print(result)  # prints 11
#######################
#Pass Function as Argument
def add(x, y):
    return x + y

def calculate(func, x, y):
    return func(x, y)

result = calculate(add, 4, 6)
print(result)  # prints 10
########################
#Return a Function as a Value
def greeting(name):
    def hello():
        return "Hello, " + name + "!"
    return hello

greet = greeting("Sanjivani")
print(greet())  # prints "Hello, Sanjivani!"
#################################
#A Python decorator is a function that takes in a function and returns it by adding some functionality.

def make_pretty(func):
    def inner():
        print("I got decorated")
        func()
    return inner


def ordinary():
    print("I am ordinary")

# Output: I am ordinary
#################################
def make_pretty(func):
    # define the inner function 
    def inner():
        # add some additional behavior to decorated function
        print("I got decorated")

        # call original function
        func()
    # return the inner function
    return inner

# define ordinary function
def ordinary():
    print("I am ordinary")
    
# decorate the ordinary function
decorated_func = make_pretty(ordinary)

# call the decorated function
decorated_func()
##################################

def make_pretty(func):

    def inner():
        print("I got decorated")
        func()
    return inner

@make_pretty
def ordinary():
    print("I am ordinary")

ordinary()  
#############################
#Decorating Functions with Parameters
def divide(a, b):
    return a/b

def smart_divide(func):
    def inner(a, b):
        print("I am going to divide", a, "and", b)
        if b == 0:
            print("Whoops! cannot divide")
            return

        return func(a, b)
    return inner

@smart_divide
def divide(a, b):
    print(a/b)

divide(2,5)

divide(2,0)
################################
#Chaining Decorators in Python
def star(func):
    def inner(*args, **kwargs):
        print("*" * 15)
        func(*args, **kwargs)
        print("*" * 15)
    return inner
#####################################
 
def percent(func):
    def inner(*args, **kwargs):
        print("%" * 15)
        func(*args, **kwargs)
        print("%" * 15)
    return inner


@star
@percent
def printer(msg):
    print(msg)

printer("Hello")
######################################
#What are *args and *kwargs? 
#Where are they appropriate?
#Solution: Both are used to pass a variable 
#number of arguments to a function. 
#The first, *args, is
#used to pass a variable length argument list

def multiply(num1,num2):
    return num1*num2
 
print("product:", multiply(2,3)) 
########################
def multiplyThreeNumbers(num1, num2, num3):
    return num1*num2*num3
 
print("product:",multiplyThreeNumbers(1, 2, 3))
#############################
def multiplyNumbers(*numbers):
    product=1
    for n in numbers:
        product*=n
    return product
 
print("product:",multiplyNumbers(4,4,4,4,4,4))
#**kwargs allows us to pass any number of 
#keyword arguments.
def makeSentence(**words):
    sentence=''
    for word in words.values():
        sentence+=word
    return sentence
 
print('Sentence:', makeSentence(a='Kwargs ',b='are ', c='awesome!'))
#In Python, these keyword arguments are passed to the program as a Python dictionary.
def whatTechTheyUse(**kwargs):
    result = []
    for key, value in kwargs.items():
        result.append("{} uses {}".format(key, value))
    return result
 
print(whatTechTheyUse(Google='Angular', Facebook='react', Microsoft='.NET'))
###############################
#A simple example of a function 
#that uses standard arguments, 
#*args and **kwargs in Python
def printingData(codeName, *args, **kwargs):
    print("I am ", codeName)
    for arg in args:
        print("I am arg: ", arg)
    for keyWord in kwargs.items():
        print("I am kwarg: ", keyWord)
 
printingData('007', 'agent', firstName='James', lastName='Bond') 
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 08:27:40 2023

@author: sumit
"""

#pre-requsite to decoators
def plus_one(number):
    num1 = number + 1
    return num1
plus_one(5)
#output: 6
#------------------------------------------

#defining functions inside the functions 
def plus_one(number):
    
    def add_one(number):
        num1 = number + 1
        return num1
    result = add_one(number)
    return result
plus_one(4)
#output:5
#----------------------------------------------
#Passing Function as Arguments
#to other function
def plus_one(number):
    result1= number + 1 
    return result1

def function_call(function):
    result=function(5)
    return result

function_call(plus_one)

#----------------------------------------------------
#Functions Returning other Functions
def hello_function(): 
    def say_hi():
        return "Hi"
    return say_hi
hello = hello_function()
hello()
#Whenever we will call hello_function directly it will return an object 
#Therefore it is required to assign it to the hello, then call hello() function 

#===============================================================
#A python decorator is a function that takes a function as an argument
# and returns a function by adding some functionalityy to it.
def say_hi():
    return "hello there!"
def uppercase_decorator(function):
    def wrapper():
        func = function()
        make_upper = func.upper()
        return make_upper 
    return wrapper 
decorate = uppercase_decorator(say_hi)
decorate()

@uppercase_decorator 
def say_hi():
    return 'hello dear!'
say_hi()
#=================================================================
#Applying multiple decorators to a single function
#However the decorator will be applied in the order that we've called them
def split_string(function):
    def wrapper():
        func = function()
        splitted_string = func.split()
        return splitted_string
    return wrapper

@split_string
@uppercase_decorator 
def say_hi():
    return 'hello dear!'
say_hi()
#--------------------------------------------------------------------
numbers=[2,5,8,11,15]
def cal_square(numbers):
    result=[]
    for i in numbers:
        result.append(i*i)
    return result 
def cal_cube(num):
    res=[]
    for i in num:
        res.append(i*i*i)
    return res

print(cal_square(numbers))
print(cal_cube(numbers))
    
#Finding how much time a processor is taking to calculate(estimation of execution) 
import time
def cal_square(numbers):
    start=time.time()
    result=[]
    for i in numbers:
        result.append(i*i)
    end=time.time()
    print((end-start)*1000)
    print(" took " + str((end-start)*1000) + "mil sec")
    return result
def cal_cube(numbers):
    start=time.time()
    result=[]
    for i in numbers:
        result.append(i*i*i)
    end=time.time()
    print((end-start)*1000)
    print(" took " + str((end-start)*1000) + "mil sec")
    return result
array = range(1,100000)
out_square = cal_square(array)
'''output:7.47227668762207
 took 7.47227668762207mil sec'''
out_cube = cal_cube(array)
'''Output:12.53652572631836
 took 12.53652572631836mil sec'''

#---------------------------------------------------------------------
import time
def time_it(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(func.__name__ +" took " + str((end-start)*1000) + "mil sec")
        return result
    return wrapper
@time_it
def cal_square(numbers):
    result = []
    for number in numbers:
        result.append(number*number)
        return result
    
@time_it
def cal_cube(numbers):
    result = []
    for number in numbers:
        result.append(number*number*number)
        return result
array = range(1,100000)
out_square = cal_square(array)
out_cube = cal_cube(array)














 



































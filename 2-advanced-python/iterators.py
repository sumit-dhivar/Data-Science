# -*- coding: utf-8 -*-
"""
Created on Fri Apr 7 15:15:47 2023

@author: sumit
"""

numbers=[1,4,6]
value=numbers.__iter__()
item1=value.__next__()
print(item1)
item2=value.__next__()
print(item2)
item3=value.__next__()
print(item3)
################################
num2=[6,8,2]
val=iter(num2)
itm1=next(val)
print(itm1)
itm2=next(val)
print(itm2)
itm3=next(val)
print(itm3)
#########################

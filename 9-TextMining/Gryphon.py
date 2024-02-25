# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 12:35:00 2024

@author: sumit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 

x = np.array(['A','B','C','D'])
y = np.array([34,12,45,32])

plt.bar(x,y,color='lightgreen' ,width=0.5)
plt.show()

arr = np.array([[[1,2,3],[4,5,6],[7,8,9]]])

print(arr)

arr1 = np.array([1,2,3,4])
arr2 = np.array([9,8,7,6])


plt.plot(arr1,arr2,label="line1")
plt.plot(arr2,arr1 , label="line2")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.legend()
plt.show()

plt.scatter(arr1,arr2)
plt.show()

val = np.array([32,45,20,6,52],index=["a","b","c","d","e"])
plt.pie(val)
plt.show()
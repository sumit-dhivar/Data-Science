# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:46:52 2023

@author: sumit
"""
#==============================NumPy Module==========================================

"""
Numpy Assignment  - 31/05/06
"""
"Write a numpy program to get numpy version "

import numpy as np
np._version_

np.show_config()

# Add function

np.info(np.add)

# transpose
# np.transpose

# test whether none of the element are zero giving True
x=np.array([1,2,3,4])

np.all(x)
#True
# IF zero present giving False 
x=np.array([0,2,3,4])

np.all(x)

# Test any of element is non zero
x=np.array([0,0,23,0])
x
np.any(x)

# test for finite set
a=np.array([1,0,np.nan,np.inf])

np.isfinite(a)

# Check for complex number
a=np.array([1+2j,1+0j])
print(a)
print("Is number is complex : ")
np.iscomplex(a)
np.isreal(a)

np.isscalar(3.1)
np.isscalar([3.1])

a=[[11,12,12],[23,34,45],[23,34,54]]

# Show the dimenstion

A=np.array(a)
A

A.ndim

# Show shape
A.shape

# Row*Col
A.size

# Accesss rows and se
A[1,2]

A[1,2]

A[1,1]

A[0,0]

' OR '

A[0][0]

# 1st row and 1st and 2nd col

A[0][0:2]

A[:1,2]

# Basic operations 

x=np.array([[2,1],[2,4]])

y=np.array([[2,1],[2,4]])

# Add
Z=x+y
Z

# Sub
Z=x-y
Z

# Mul
Z=x*y
Z

"or "

Z=np.dot(x,y)
Z

# Calculate the sine of Z

np.sin(Z)

# Calucate transpose of matrix

c=np.array([[1,1],[2,2],[3,3]])
c

c.T

#-------------------------------------------------------------------
import numpy as np 
p = [[1,0],[0,1]] 
q = [[1,2],[3,4]] 
print("Original Matrix:- ") 
print(p) 
print(q) 
result = np.dot(p,q)
print(result)
# [[1 2]
#  [3 4]]

#Outer product 
p = [[1,0],[0,1]] 
q = [[1,2],[3,4]] 
print("Original Matrix:- ") 
print(p) 
print(q) 
result = np.outer(p,q)
print(result)
# [[1 2 3 4]
#  [0 0 0 0]
#  [0 0 0 0]
#  [1 2 3 4]]

#Cross Product 
res = np.cross(p,q)
print(res)
#[ 2 -3]
res = np.cross(q,p)
print(res)
#[-2  3]

#Write a program to compute the determinant of a given square 
import numpy as np 
from numpy import linalg as LA 
a = np.array([[1,0],[1,2]]) 
print("Original 2-D array ")
print(a)
print("Determinant of the said 2-D array:- ")
print(np.linalg.det(a))

#Write a program to compute the eigenvalues and right eigenvector 

m = np.mat('3 -2:1 0') 
print("Original Matrix: - ",m)
w,v = np.linalg.eig(m)
print("EigenVector of the said matrix: ",w)
print("EigenValues of the said matrix: ",v)

#Inverse of a given matrix 
m = np.array([[1,2],[3,4]])
print("Original Matrix : ",m)
res = np.linalg.inv(m)
print("Inverse of matrix is ",res)

#Write a Numpy program to generate five random numbers from the normal distributuin 
import numpy as np 
x = np.random.normal(size = 15)
print(x)

#Write a progrma to generate six random integers between 10 and 30
import random 
x = np.random.randint(low = 10,high=30,size = 6)
print(x)

#Write a NumPy program to create a 3x3x3 array with random values 
x = np.random.random((3,3,3))
print(x)

#Write a NumPy program to create a 5x5 array with random values and find the minimum and maximum values
import numpy as np 
x = np.random.random((5,5))
print("Original matrix is ",x)
xmin,xmax = x.min(),x.max()
print("Max and Min values are ",xmax,xmin)












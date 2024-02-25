# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 08:17:59 2023

@author: sumit
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
print("Hellooooo SD")

# Write a NumPy program to get the minimum  and maximum value


x = np.arange(4).reshape((2, 2))
print("\noriginal array: ")
print(x)
# [[0 1]
# [2 3]]

print("\nMaximum value along the second axis:")
print(np.amax(x, 1))
# [1 3]

print("Minimum value along the second axis:")
print(np.amin(x, 1))
# [0 2]
# 3
# Write a Python program to draw a line with suitable label
X = range(1, 50)
Y = [value * 3 for value in X]
print("Values of X:")
print(range(1, 50))
print("Values of Y (thrice of X):")
print(Y)
plt.plot(X, Y)
plt.title("This is Line Graph!")
plt.xlabel("This is a X-axis.")
plt.ylabel("This is a Y-axis.")
####################################
# python program to draw line using given axis value

# x axis values
X = [1, 2]

# y axis values
y = [2, 4]

# Plot lines and/or markers to the Axes.
plt.plot(x, y)

# Set the x axis label of the current axis.
# Set the y axis label of the current axis.
plt.xlabel('x - axis')
plt.ylabel('y - axis')

# Set a title
plt.title("Sample Grapgh")
plt.show()
#########################################################
# write a python program to draw line chart of report
# between october 3, 2016 to october 7, 2016
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("C:/1-Python/FData.csv")
df.plot()
plt.show()
########################################################
#ploting a gragh of all student with their attendence
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("C:/1-Python/DS_Attendance.csv")
df.plot()
plt.show()
##############################################################
import matplotlib.pyplot as plt
import pandas as pd
df = pd.read_csv("C:/1-Python/DS_attendance(1).csv")
df.plot()
plt.show()
#####################################################

#Write a Python program to plot legends ,diff
import matplotlib.pyplot as plt
# line 1 points
x1=[10,20,30]
y1= [20,40,10]
#line 2 plot
x2=[40,50,60]
y2=[70,80,90]
# Set the x axis label of the current axis. 
plt.xlabel('x axis')
# Set the y axis label of the current axis.
plt.ylabel('y axis')

# Set a title
plt.title("Two or more lines with different widths and colors.")

plt.plot(x1,y1, color='blue', linewidth=3, label='linel-width-3')

plt.plot(x2,y2, color='red', linewidth=5, label='line2-width-5') 
#show a legend on the plot
plt.legend()
plt.show()
###########################################################
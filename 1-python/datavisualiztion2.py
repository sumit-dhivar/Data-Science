# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 07:45:47 2023

@author: sumit
"""

#Write a program to plot two or more lines with legends, different width and color 
import matplotlib.pyplot as plt 
# line 1 points 
x1 = [10,20,30] 
y1 = [20,40,10]

#line2 points 
x2 = [10,20,30]
y2 = [40,10,30]

#Set the x axis label of the current axis 
plt.xlabel('x-axis') 
#Set the y axis label of the current axis 
plt.ylabel('y-axis')
#Set a title 
plt.title('Two or more lines with different widths and colors with suitable')
#Display the figure 
plt.plot(x1,y1,color='blue',linewidth = 3, label = 'line-width-3')
plt.plot(x2,y2,color='red',linewidth = 5,label = 'line2 - width - 5')

#Show a legend on the plot 
plt.legend()

#===================================================================

import matplotlib.pyplot as plt 
# line 1 points 
x1 = [10,20,30] 
y1 = [20,40,10]

#line2 points 
x2 = [10,20,30]
y2 = [40,10,30]

#Set the x axis label of the current axis 
plt.xlabel('x-axis') 
#Set the y axis label of the current axis 
plt.ylabel('y-axis')
#Set a title 
plt.title('Two or more lines with different widths and colors with suitable')
#Display the figure 
plt.plot(x1,y1,color='blue',linewidth = 3, label = 'line1-dotted',linestyle='dotted')
plt.plot(x2,y2,color='red',linewidth = 5,label = 'line2-dashed',linestyle='dashed')

#Show a legend on the plot 
plt.legend()

#Function to show plot 
plt.show()

#Plot two or more lines and set the line markers 
import matplotlib.pyplot 
# x axis values 
x = [1,4,5,6,7] 
# y axis values 
y = [2,6,3,6,3]
#plotting the points 
plt.plot(x,y,color='red',linestyle='dashdot',linewidth = 3,marker='o',markerfacecolor = 'green',markersize = 15)
#Set y-limits of the current axes
plt.ylim(1,8)
#Set x-limits of the current axes
plt.xlim(1,8)

#naming the x axis 
plt.xlabel('x-axis')
#naming the y axis
plt.ylabel('y-axis')

#giving a title to my graph 
plt.title("Display the marker ")
# function to show the plot 
plt.show()

#=============================================================
import numpy as np 
import matplotlib.pyplot 
#Sampled time at 200ms intervals 
t = np.arange(0.,5.,0.2)

#green dashess, blue squares and red triangles 
plt.plot(t,t,'g--',t,t**2,'bs',t,t**3,'r^')
plt.show()

#to display a bar chart of the popularity of programming language 
import matplotlib.pyplot as plt 
x = ['Java','Python','PHP','Javascript', ' C#','C++']
popularity = [22.2,17.6,8.8,8,7.7,6.7]
#Enumerate is a built-in function in python that allows you to keep track of the number of iterations (loops) in a loop. 
x_pos = [i for i,j in enumerate(x)]
plt.bar(x_pos,popularity,color='blue')
plt.xlabel("Languages")
plt.ylabel("Popularity")
plt.title("Popularity of Prog. Languages\n" + "Worldwide, Oct 2017 compared to a ")
plt.xticks(x_pos,x)#To display graph horizontally
plt.show()
#===================================================================
import matplotlib.pyplot as plt 
x = ['Java','Python','PHP','Javascript', ' C#','C++']
popularity = [22.2,17.6,8.8,8,7.7,6.7]
#Enumerate is a built-in function in python that allows you to keep track of the number of iterations (loops) in a loop. 
x_pos = [i for i,j in enumerate(x)]
plt.barh(x_pos,popularity,color='blue')
plt.xlabel("Languages")
plt.ylabel("Popularity")
plt.title("Popularity of Prog. Languages\n" + "Worldwide, Oct 2017 compared to a ")
plt.yticks(x_pos,x)#To display graph horizontally
plt.show()

#====================================================================
import numpy as np 
import matplotlib.pyplot as plt 

#data to plot 
n_groups = 5 
men_means = (22,30,33,30,26)
women_means = (25,32,30,35,29)

#create plot 
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.35 
opacity = 0.8 
rects1 = plt.bar(index,men_means,bar_width,alpha=opacity,color='g',label='Men')
rects2 = plt.bar(index+bar_width,women_means,bar_width,alpha=opacity,color='r',label='Women')
plt.xlabel('Person')
plt.ylabel('Scores')
plt.title('scores by person')
plt.xticks(index + bar_width,('g1','g2','g3','g4','g5'))
plt.legend()
plt.tight_layout()
plt.show()

# -*- coding: utf-8 -*-
"""
Spyder Editor


"""

#Data Processing


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("D:/ANACONDA/Data Science/CSV Files/ethnic diversity.csv")

df

#Now Lets Perform EDA

df.shape
#(310, 13) 

df.columns
#['Employee_Name', 'EmpID', 'Position', 'State', 'Zip', 'Sex',
       # 'MaritalDesc', 'CitizenDesc', 'EmploymentStatus', 'Department',
       # 'Salaries', 'age', 'Race'] 
 
       
#Here we have plotted the 2-D Scatter Plot 
df.plot(kind='scatter', x='EmpID', y='Zip') ;
plt.show()


#Let us check the data types 
df.dtypes

#Let's convert Salaries data type to int from float
df.Salaries = df.Salaries.astype(int)
df.dtypes

#Let's convert Salaries data type to float from int
df.Salaries = df.Salaries.astype(float)
df.dtypes


#Let's convert age data type to float from int
df.age = df.age.astype(float)
df.dtypes
#========================================================================

df_new = pd.read_csv("D:/ANACONDA/Data Science/CSV Files/education.csv")
df_new

duplicate = df_new.duplicated()
#Output of this function is a single column
#If there is a duplicate records output-True 
#If there is no duplicate records output- False 
#Series will be created
duplicate

sum(duplicate)#The sum will return the count of only  the true values
#0
df_new1 = pd.read_csv("D:/ANACONDA/Data Science/CSV Files/mtcars_dup.csv")
duplicate1 = df_new1.duplicated()
duplicate1
sum(duplicate1)

#three duplicate records found at 17,23,27 
#we need to drop these records, row 17 is duplicate of 2
df_new1 = pd.read_csv("D:/ANACONDA/Data Science/CSV Files/mtcars_dup.csv")
duplicate1 = df_new1.duplicated()
duplicate1
sum(duplicate1)


df_new2 = df_new1.drop_duplicates()
duplicate2 = df_new2.duplicated()
duplicate2
sum(duplicate2)

#Outlier treatment 
import pandas as pd
import seaborn as sns 
df = pd.read_csv('D:/ANACONDA/Data Science/CSV Files/ethnic diversity.csv')

df.var

#lets find the outliers with the help of box plot
sns.boxplot(df.Salaries)

#From box plot we can infere that there are noo any outlieres 

sns.boxplot(df.age)

#Lets calculate IQR 
IQR = df.Salaries.quantile(0.75) - df.Salaries.quantile(0.25)

lower_limit = df.Salaries.quantile(0.25)-1.5*IQR

upper_limit = df.Salaries.quantile(0.75)+1.5*IQR

#If we have lower limit in -ive then we have to make it 0 by just clicking it in
# the variable explorer


#-----------------------------------------------------------------
#Trimming
import numpy as np 
outliers_df = np.where(df.Salaries>upper_limit,True,np.where(df.Salaries<lower_limit,True,False))


df_trimmed = df.loc[~outliers_df]
df.shape
#(310, 13)
df_trimmed.shape
#(306, 13)

#here we can see that we have  4 outliers(23,63 ,74,216,)


#Replace ment Technique 
df_replaced = pd.DataFrame(np.where(df.Salaries > upper_limit, upper_limit, np.where(
    df.Salaries < lower_limit, lower_limit, df.Salaries)))


#------------------------------------------------------------------------------


#Winsorizer 
import pandas as pd
import seaborn as sns
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr',
                  tail='both',
                  fold=1.5,
                  variables=['Salaries']
                  )


df = pd.read_csv('D:/ANACONDA/Data Science/CSV Files/ethnic diversity.csv')

#Copy Winsorizer and pasre in Help Tab of top righ window study the method 

df_t = winsor.fit_transform(df[['Salaries']])
sns.boxplot(df['Salaries'])
sns.boxplot(df_t['Salaries'])


#-----------------------------------Assignment--------------------------------------

"""
The Boston Housing Dataset is a derived from information collected by the U.S. Census Service concerning housing in the area of Boston MA. The following describes the dataset columns:

CRIM - per capita crime rate by town
ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
INDUS - proportion of non-retail business acres per town.
CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
NOX - nitric oxides concentration (parts per 10 million)
**RM - average number of rooms per dwelling (cont.->disc)
AGE - proportion of owner-occupied units built prior to 1940
DIS - weighted distances to five Boston employment centres
RAD - index of accessibility to radial highways
TAX - full-value property-tax rate per $10,000
PTRATIO - pupil-teacher ratio by town
B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
LSTAT - % lower status of the population
MEDV - Median value of owner-occupied homes in $1000's

Buisness Objective:- 

Minimize: 
    
Maximize: 
    
Buisness Constraint:-

Data Dictionary:-
crim       continous
zn         continous
indus      continous
chas       continous
nox        continous
rm         continous
age        continous
dis        continous
rad        ordinal
tax        discrete
ptratio    continous
black      continous
lstat      continous
medv       continous
"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df = pd.read_csv('D:/ANACONDA/Data Science/CSV Files/boston_data.csv',skiprows=[0]) 

#lets find the shape of the ddata set 
df.shape
#So here we have 404 rows and 14 columns 
#Lets find which columns are there  
df.columns

















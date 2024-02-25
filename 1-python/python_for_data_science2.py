# -*- coding: utf-8 -*-
"""
Created on Tue Apr  12 16:26:24 2023

@author: sumit
"""

#What is Pandas DataFrame?
#pandas DataFrame is a Two-Dimensional data structure,
#an immutable, heterogeneous tabular 
#data structure with labeled axes rows, and columns.

#DataFrame Features
#DataFrames support named rows & columns
# (you can also provide names to rows)
#Supports heterogeneous collections of data.
#DataFrame labeled axes (rows and columns).
#Installing Pandas
#step-1 go to anaconda navigator
#step-2 select environment tab
#step-3 by default it will be base terminal
#step-4 on base terminal-pip install pandas
# Or conda install pandas
######################################
#Upgrade Pandas to Latest or Specific Version
#on base terminal write
#conda install --upgrade pandas

#upgrade to specific version
#conda update pandas==1.5.3
##############################
#To check the version of pandas
import pandas as pd
pd.__version__
#################################
#Create using Constructor
# Create pandas DataFrame from List
import pandas as pd
technologies = [ ["Spark",20000, "30days"], 
                 ["pandas",20000, "40days"] 
               ]
df=pd.DataFrame(technologies)
print(df)

#Since we have not given labels to columns and 
#indexes, DataFrame by default assigns 
#incremental sequence numbers as labels 
#to both rows and columns, these are called Index.
# Add Column & Row Labels to the DataFrame
column_names=["Courses","Fee","Duration"]
row_label=["a","b"]
df=pd.DataFrame(technologies,columns=column_names,index=row_label)
print(df)
###########################
df.dtypes 
#####################
#You can also assign custom 
#data types to columns.
# set custom types to DataFrame
import pandas as pd
technologies = {
    'Courses':["Spark","PySpark","Hadoop","Python","pandas","Oracle","Java"],
    'Fee' :[20000,25000,26000,22000,24000,21000,22000],
    'Duration ':['30day','40days','35days', '40days','60days','50days','55days'],
    'Discount':[11.8,23.7,13.4,15.7,12.5,25.4,18.4]
    }
df = pd.DataFrame(technologies)
print(df.dtypes)
# Convert all types to best possible types
df2=df.convert_dtypes()
print(df2.dtypes)
# Change All Columns to Same type
df = df.astype(str)
print(df.dtypes)
# Change Type For One or Multiple Columns
df = df.astype({"Fee": int, "Discount": float})
print(df.dtypes)
# Convert Data Type for All Columns in a List
df = pd.DataFrame(technologies)
df.dtypes
cols = ['Fee', 'Discount']
df[cols] = df[cols].astype('float')
df.dtypes
#Ignores error
df = df.astype({"Courses": int},errors='ignore')
df.dtypes
# Generates error
df = df.astype({"Courses": int},errors='raise')
# Converts feed column to numeric type
df = df.astype(str)
print(df.dtypes)
df['Discount'] = pd.to_numeric(df['Discount'])
df.dtypes


###########################

# Create DataFrame from Dictionary
technologies = {
    'Courses':["Spark","PySpark","Hadoop"],
    'Fee' :[20000,25000,26000],
    'Duration':['30day','40days','35days'],
    'Discount':[1000,2300,1500]
              }
df = pd.DataFrame(technologies)
df
##############################
#convert dataframe to csv
df.to_csv('data_file.csv')
##################################
#Create DataFrame From CSV File

df = pd.read_csv('data_file.csv')
#################################
#Pandas DataFrame – Basic Operations
# Create DataFrame with None/Null to work with examples
import pandas as pd
import numpy as np
technologies   = ({
    'Courses':["Spark","PySpark","Hadoop","Python","Pandas",None,"Spark","Python"],
    'Fee' :[22000,25000,23000,24000,np.nan,25000,25000,22000],
    'Duration':['30day','50days','55days','40days','60days','35day','','50days'],
    'Discount':[1000,2300,1000,1200,2500,1300,1400,1600]
          })
row_labels=['r0','r1','r2','r3','r4','r5','r6','r7']
df = pd.DataFrame(technologies, index=row_labels)
print(df)
###################################
#DataFrame properties
df.shape
#(8, 4)
df.size
#32
df.columns
df.columns.values
df.index
df.dtypes
#############################
#Accessing one column contents
df['Fee']
##Accessing two columns contents
df[['Fee','Duration']]
#select certain rows and assign it to another dataframe
df2=df[6:]
df2
#########################
#accessing certain cell from column 'Duration'
df['Duration'][3]
#subtracting specific value from a column
df['Fee'] = df['Fee'] - 500
df['Fee']
#Pandas to Manipulate DataFrame
#Describe DataFrame
# Describe DataFrame for all numberic columns
df.describe()
#It will show 5 number summary
########################################
#rename() – Renames pandas DataFrame columns
df = pd.DataFrame(technologies, index=row_labels)

# Assign new header by setting new column names.
df.columns=['A','B','C','D']
df
##############################
# Rename Column Names using rename() method
df = pd.DataFrame(technologies, index=row_labels)
df.columns=['A','B','C','D']
df2 = df.rename({'A': 'c1', 'B': 'c2'}, axis=1)
df2 = df.rename({'C': 'c3', 'D': 'c4'}, axis='columns')
df2 = df.rename(columns={'A': 'c1', 'B': 'c2'})
###################################
#Drop DataFrame Rows and Columns
df = pd.DataFrame(technologies, index=row_labels)

# Drop rows by labels
df1 = df.drop(['r1','r2'])
df1
# Delete Rows by position/index
df1=df.drop(df.index[1])
df1
df1=df.drop(df.index[[1,3]])
df1
# Delete Rows by Index Range
df1=df.drop(df.index[2:])

# When you have default indexs for rows
df = pd.DataFrame(technologies)
df1 = df.drop(0)
df1
df = pd.DataFrame(technologies)
df1 = df.drop([0, 3])#it will delete row0 n row3
df1 = df.drop(range(0,2))#it will delete 0 and 1
#################################
##################################
import pandas as pd
technologies = ({
    'Courses':["Spark","PySpark","Hadoop","Python","pandas","Oracle","Java"],
    'Fee' :[20000,25000,26000,22000,24000,21000,22000],
    'Duration':['30day', '40days' ,'35days', '40days', '60days', '50days', '55days']
              })
df = pd.DataFrame(technologies)
print(df)
#Drop Column by Name
# Drops 'Fee' column
# A DataFrame object has two axes: “axis 0” and “axis 1”. 
#“axis 0” represents rows and “axis 1” represents columns.
df2=df.drop(["Fee"], axis = 1)
print(df2)

# Explicitly using parameter name 'labels'
df2=df.drop(labels=["Fee"], axis = 1)

# Alternatively you can also use columns instead of labels.
df2=df.drop(columns=["Fee"], axis = 1)
################################
# Drop column by index.
print(df.drop(df.columns[1], axis = 1))

df = pd.DataFrame(technologies)

# using inplace=True
df.drop(df.columns[[2]], axis = 1, inplace=True)
print(df)
#####################################
df = pd.DataFrame(technologies)
#Drop Two or More Columns By Label Name
df2=df.drop(["Courses", "Fee"], axis = 1)
print(df2)
#######################################
#Drop Two or More Columns by Index
df = pd.DataFrame(technologies)

df2=df.drop(df.columns[[0,1]], axis = 1)
print(df2)
###################################
#Drop Columns from List of Columns
df = pd.DataFrame(technologies)
lisCol = ["Courses","Fee"]
df2=df.drop(lisCol, axis = 1)
print(df2)
#############################
#Remove columns From DataFrame inplace
df.drop(df.columns[1], axis = 1, inplace=True)
df
# using inplace=True
########################################
########################################
##Pandas Select Rows by Index (Position/Label)
import pandas as pd
import numpy as np
technologies   = ({
    'Courses':["Spark","PySpark","Hadoop","Python","Pandas",None,"Spark","Python"],
    'Fee' :[22000,25000,23000,24000,np.nan,25000,25000,22000],
    'Duration':['30day','50days','55days','40days','60days','35day','','50days'],
    'Discount':[1000,2300,1000,1200,2500,1300,1400,1600]
          })
row_labels=['r0','r1','r2','r3','r4','r5','r6','r7']
df = pd.DataFrame(technologies, index=row_labels)
print(df)

#df.iloc[startrow:endrow, startcolumn:endcolumn]

df = pd.DataFrame(technologies, index=row_labels)

# Below are quick example
df2=df.iloc[:, 0:2]
df2
#This line uses the slicing operator to get DataFrame
# items by index.
# The first slice [:] indicates to return all rows. 
#The second slice specifies that only columns 
#between 0 and 2 (excluding 2) should be returned.

df2=df.iloc[0:2, :]
df2
#In this case, the first slice [0:2] is 
#requesting only rows 0 through 1of the DataFrame.
#The second slice [:] indicates that all columns are required.

#Slicing Specific Rows and Columns using iloc attribute
df3=df.iloc[1:2, 1:3]
df3
#Another example
df3=df.iloc[:, 1:3]
df3
#The second operator [1:3] yields columns 1 and 3 only.
# Select Rows by Integer Index
df2 = df.iloc[2]     # Select Row by Index
df2




df2 = df.iloc[[2,3,6]]    # Select Rows by Index List
df2 = df.iloc[1:5]   # Select Rows by Integer Index Range
df2 = df.iloc[:1]    # Select First Row
df2 = df.iloc[:3]    # Select First 3 Rows
df2 = df.iloc[-1:]   # Select Last Row
df2 = df.iloc[-3:]   # Select Last 3 Row
df2 = df.iloc[::2]   # Selects alternate rows

# Select Rows by Index Labels
df2 = df.loc['r2']          # Select Row by Index Label
df2 = df.loc[['r2','r3','r6']]    # Select Rows by Index Label List
df2 = df.loc['r1':'r5']     # Select Rows by Label Index Range
df2 = df.loc['r1':'r5']     # Select Rows by Label Index Range
df2 = df.loc['r1':'r5':2]   # Select Alternate Rows with in Index La
##########################################
import pandas as pd
import numpy as np
technologies = {
    'Courses':["Spark","PySpark","Hadoop","Python","pandas","Oracle","Java"],
    'Fee' :[20000,25000,26000,22000,24000,21000,22000],
    'Duration':['30days','40days','35days','40days',np.nan,None,'55days'],
    'Discount':[1000,2300,1500,1200,2500,2100,2000]
               }
index_labels=['r0','r1','r2','r3','r4','r5','r6']
df = pd.DataFrame(technologies,index=index_labels)
print(df)
#Select Rows by Index using Pandas iloc[]
#Select Row by Integer Index

print(df.iloc[2])
# Outputs
#Courses     Hadoop
#Fee          26000
#Duration    35days
#Discount      1500
#Name: r3, dtype: object
###################################
#Get Multiple Rows by Index List
# Select Rows by Index List
print(df.iloc[[2,3,6]])
# Outputs
#   Courses    Fee Duration  Discount
#r2  Hadoop  26000   35days      1500
#r3  Python  22000   40days      1200
#r6    Java  22000   55days      2000   2000
#######################################
#Get DataFrame Rows by Index Range
# Select Rows by Integer Index Range
print(df.iloc[1:5])
# Output
#Courses    Fee Duration  Discount
#r1  PySpark  25000   40days      2300
#r2   Hadoop  26000   35days      1500
#r3   Python  22000   40days      1200
#r4   pandas  24000      NaN      2500
############################
# Select First Row by Index
print(df.iloc[:1])
# Outputs
#   Courses    Fee Duration  Discount
#r0   Spark  20000   30days      1000

# Select First 3 Rows
print(df.iloc[:3])
# Outputs
#    Courses    Fee Duration  Discount
#r0    Spark  20000   30days      1000
#r1  PySpark  25000   40days      2300
#r2   Hadoop  26000   35days      1500

# Select Last Row by Index
print(df.iloc[-1:])
# Outputs
#    Courses    Fee Duration  Discount
#r6    Java  22000   55days      2000
# Select Last 3 Row
print(df.iloc[-3:])
# Outputs
#   Courses    Fee Duration  Discount
#r4  pandas  24000      NaN      2500
#r5  Oracle  21000     None      2100
#r6    Java  22000   55days      2000
#############################
#####################################
#Pandas Select Columns by Name or Index
# By using df[] Notation
df2=df['Courses']
## select multile columns
df2 = df[["Courses","Fee","Duration"]] 

# Using loc[] to take column slices
#loc[] syntax to slice columns
#df.loc[:,start:stop:step]
## Selecte multiple columns
df2 = df.loc[:, ["Courses","Fee","Duration"]]
# Select Random columns 
df2 = df.loc[:, ["Courses","Fee","Discount"]]
# Select columns between two columns 
df2 = df.loc[:,'Fee':'Discount'] 
## Select columns by range
df2 = df.loc[:,'Duration':] 
# Select columns by range 
#All the columns upto 'Duration'
df2 = df.loc[:,:'Duration']  
## Select every alternate column
df2 = df.loc[:,::2]          
###############################
#Pandas iloc[] to Select Column by Index or Position
#Select Multiple Columns by Index Position
# Selected by column position
df2 = df.iloc[:,[1,2,3]]
df2
#Fee Duration  Discount
#r0  20000   30days      1000
#r1  25000   40days      2300
#r2  26000   35days      1500
#r3  22000   40days      1200
#r4  24000      NaN      2500
#r5  21000     None      2100
#r6  22000   55days      2000
#########################
# Select between indexes 1 and 4 (2,3,4)
df2 = df.iloc[:,1:4]
df2
#Returns
#     Fee Duration  Discount
#0  20000   30days      1000
#1  25000   40days      2300
##########################
# Select From 3rd to end
df2 = df.iloc[:,2:]
df2
#Returns
#  Duration  Discount   Tutor
#0   30days      1000  Michel
#1   40days      2300     Sam
########################
 #Select First Two Columns
df2 = df.iloc[:,:2]
df2
#Returns
#   Courses    Fee
#0    Spark  20000
#1  PySpark  25000
################################
################################
#Pandas.DataFrame.query() by Examples
# Query all rows with Courses equals 'Spark'
df2=df.query("Courses == 'Spark'")
print(df2)
############################
# not equals condition
df2=df.query("Courses != 'Spark'")
df2
#############################
##############################

#Pandas Add Column to DataFrame
import pandas as pd
import numpy as np

technologies= {
    'Courses':["Spark","PySpark","Hadoop","Python","Pandas"],
    'Fee' :[22000,25000,23000,24000,26000],
    'Discount':[1000,2300,1000,1200,2500]
          }

df = pd.DataFrame(technologies)
print(df)
####################################
#Pandas Add Column to DataFrame
# Add new column to the DataFrame
tutors = ['Ram', 'sham', 'Ghansham', 'Ganesh', 'Ramesh']
df2 = df.assign(TutorsAssigned=tutors)
print(df2)
############################
# Add multiple columns to the DataFrame
MNCCompanies = ['TATA','HCL','Infosys','Google','Amazon']
df2 = df.assign(MNCComp = MNCCompanies,TutorsAssigned=tutors )
df2
############################
# Derive New Column from Existing Column
df = pd.DataFrame(technologies)
df2 = df.assign(Discount_Percent=lambda x: x.Fee * x.Discount / 100)
print(df2)
############################
#Append Column to Existing Pandas DataFrame
# Add New column to the existing DataFrame
df = pd.DataFrame(technologies)
df["MNCCompanies"] = MNCCompanies
print(df)
#############################
# Add new column at the specific position
df = pd.DataFrame(technologies)
df.insert(0,'Tutors', tutors )
print(df)
######################
######################
#Pandas Rename Column with Examples
import pandas as pd
technologies = ({
  'Courses':["Spark","PySpark","Hadoop","Python","pandas","Oracle","Java"],
  'Fee' :[20000,25000,26000,22000,24000,21000,22000],
  'Duration':['30day', '40days' ,'35days', '40days', '60days', '50days', '55days']
              })
df = pd.DataFrame(technologies)
print(df.columns)
#Pandas Rename Column Name
# Rename a Single Column 
df2=df.rename(columns = {'Courses':'Courses_List'})
print(df2.columns)
#Alternatively, you can also write
# the above statement by using 
#axis=1 or axis='columns'
# Alternatively you can write above using axis
df2=df.rename({'Courses':'Courses_List'}, axis=1)
df2=df.rename({'Courses':'Courses_List'}, axis='columns')
##################################
#In order to change columns on the existing DataFrame
# without copying to the new DataFrame, 
#you have to use inplace=True.
# Replace existing DataFrame (inplace). This returns None.
df.rename({'Courses':'Courses_List'}, axis='columns', inplace=True)
print(df.columns)
####################################33
#Rename Multiple Columns

df.rename(columns = {'Courses':'Courses_List','Fee':'Courses_Fee', 
   'Duration':'Courses_Duration'}, inplace = True)
print(df.columns)
df.columns
################################
#Rename Columns with a List

column_names = ['Courses','Fee','Duration']
df.columns = column_names
print(df.columns)
##################################
 #Rename multiple columns with inplace
df.rename(columns = {'Courses':'Courses_List','Fee':'Courses_Fee', 
   'Duration':'Courses_Duration'}, inplace = True)
print(df.columns)
#######################################
#######################################
#Pandas Drop Rows From DataFrame
import pandas as pd
import numpy as np

technologies = {
    'Courses':["Spark","PySpark","Hadoop","Python"],
    'Fee' :[20000,25000,26000,22000],
    'Duration':['30day','40days',np.nan, None],
    'Discount':[1000,2300,1500,1200]
               }

indexes=['r1','r2','r3','r4']
df = pd.DataFrame(technologies,index=indexes)
print(df)
###################################
#pandas Drop Rows From DataFrame Examples
# Drop rows by Index Label
df = pd.DataFrame(technologies,index=indexes)
df1 = df.drop(['r1','r2'])
print(df1)
#######################
# Delete Rows by Index Labels
df1 = df.drop(index=['r1','r2'])
df1
###############################
# Delete Rows by Index Labels & axis
df1 = df.drop(labels=['r1','r2'])
df1 = df.drop(labels=['r1','r2'],axis=0)
#####################################
# Delete Rows by Index numbers
df = pd.DataFrame(technologies,index=indexes)
df1=df.drop(df.index[[1,3]])
print(df1)
############################
# Delete Rows by Index Range
df = pd.DataFrame(technologies,index=indexes)
df1=df.drop(df.index[2:])
print(df1)
################################
#Delete Rows when you have Default Indexes
# Remove rows when you have default index.
df = pd.DataFrame(technologies)
df1 = df.drop(0)
df3 = df.drop([0, 3])
df4 = df.drop(range(0,2))
#####################################
#Remove DataFrame Rows inplace
# Delete Rows inplace
df = pd.DataFrame(technologies,index=indexes)
df.drop(['r1','r2'],inplace=True)
print(df)
##################################
#Drop Rows that has NaN/None/Null Values
# Delete rows with Nan, None & Null Values
df = pd.DataFrame(technologies,index=indexes)
df2=df.dropna()
print(df2)
#####################################
#Remove Rows by Slicing DataFrame
df2=df[4:]     # Returns rows from 4th row
df2=df[1:-1]   # Removes first and last row
df2=df[2:4]    # Return rows between 2 and 4
####################################
#Change All Columns to Same type in Pandas
#df.astype(str) converts all columns of Pandas DataFrame to string type.

df = df.astype(str)
print(df.dtypes)
###############################
#Change Type For One or Multiple Columns in Pandas
# Change Type For One or Multiple Columns
df = df.astype({"Fee": int, "Discount": float})
print(df.dtypes)
##############################
#Convert Data Type for All Columns in a List
 
df = pd.DataFrame(technologies)
cols = ['Fee', 'Discount']
df[cols] = df[cols].astype('float')

# By using a loop
for col in ['Fee', 'Discount']:
    df[col] = df[col].astype('float')

########################################
#Raise or Ignore Error when Convert Column type Fails

df = df.astype({"Courses": int},errors='ignore')

# Generates error
df = df.astype({"Courses": int},errors='raise')
####################################
#Using DataFrame.to_numeric() to Convert Numeric Types
# Converts feed column to numeric type
df['Fee'] = pd.to_numeric(df['Fee'])
################################
#Convert multiple Numeric Types using apply() Method
# Convert Fee and Discount to numeric types
df = pd.DataFrame(technologies)
df[['Fee', 'Discount']] =df [['Fee', 'Discount']].apply(pd.to_numeric)
print(df.dtypes)
###################################
#Quick Examples of Get the Number of Rows in DataFrame
rows_count = len(df.index)
rows_count
rows_count = len(df.axes[0])
rows_count
##################################
df = pd.DataFrame(technologies)
row_count = df.shape[0]  # Returns number of rows
col_count = df.shape[1]  # Returns number of columns
print(row_count)
#Outputs - 4
##################################
#pandas Apply Function to a Column
# Below are quick examples
# Using Dataframe.apply() to apply function add column

import pandas as pd
import numpy as np
data = [(3,5,7), (2,4,6),(5,8,9)]
df = pd.DataFrame(data, columns = ['A','B','C'])
print(df)


def add_3(x):
   return x+3
df2 = df.apply(add_3)
df2
##########################
# Using apply function single column
def add_4(x):
   return x+4
df["B"] = df["B"].apply(add_4)
df["B"]
# Apply to multiple columns
df[['A','B']] = df[['A','B']].apply(add_3)

# apply a lambda function to each column
df2 = df.apply(lambda x : x + 10)
##################################
# apply() function on selected list of multiple columns
df = pd.DataFrame(data, columns = ['A','B','C'])
df[['A','B']] = df[['A','B']].apply(add_3)
print(df)
##############################
#Apply Lambda Function to Each Column

df2 = df.apply(lambda x : x + 10)
print(df2)
#############################
#Apply Lambda Function to Single Column
# Using Dataframe.apply() and lambda function
df["A"] = df["A"].apply(lambda x: x-2)
print(df)
################################
#Using pandas.DataFrame.transform() to Apply Function Column
# Using DataFrame.transform() 
def add_2(x):
    return x+2
df = df.transform(add_2)
print(df)
#############################
#Using pandas.DataFrame.map() to Single Column

df['A'] = df['A'].map(lambda A: A/2.)
print(df)
######################
#Using Numpy function on single Column
# Using Dataframe.apply() & [] operator
df['A'] = df['A'].apply(np.square)
print(df)
#########################
#Using NumPy.square() Method
# Using numpy.square() and [] operator
df['A'] = np.square(df['A'])
print(df)
#################
#Pandas groupby()  With Examples 
import pandas as pd
technologies   = ({
    'Courses':["Spark","PySpark","Hadoop","Python","Pandas","Hadoop","Spark","Python","NA"],
    'Fee' :[22000,25000,23000,24000,26000,25000,25000,22000,1500],
    'Duration':['30days','50days','55days','40days','60days','35days','30days','50days','40days'],
    'Discount':[1000,2300,1000,1200,2500,None,1400,1600,0]
          })
df = pd.DataFrame(technologies)
print(df)

# Use groupby() to compute the sum
df2 =df.groupby(['Courses']).sum()
print(df2)
######################
# Group by multiple columns
df2 =df.groupby(['Courses', 'Duration']).sum()
print(df2)
##########################
#Add Index to the grouped data
# Add Row Index to the group by result
df2 = df.groupby(['Courses','Duration']).sum().reset_index()
print(df2)
##########################
# Group by on multiple columns
df2 =df.groupby(['Courses', 'Duration']).sum()
print(df2)
##############################
#Pandas Get Column Names from DataFrame
import pandas as pd
import numpy as np

technologies= {
    'Courses':["Spark","PySpark","Hadoop","Python","Pandas"],
    'Fee' :[22000,25000,23000,24000,26000],
    'Duration':['30days','50days','30days', None,np.nan],
    'Discount':[1000,2300,1000,1200,2500]
          }
df = pd.DataFrame(technologies)
print(df)

# Get the list of all column names from headers
column_headers = list(df.columns.values)
print("The Column Header :", column_headers)
###########################
#Using list(df) to get the column headers as a list
column_headers = list(df.columns)
column_headers
#Using list(df) to get the list of all Column Names
column_headers = list(df)
column_headers
############################
############################
#Pandas Shuffle DataFrame Rows 
import pandas as pd
technologies = {
    'Courses':["Spark","PySpark","Hadoop","Python","pandas","Oracle","Java"],
    'Fee' :[20000,25000,26000,22000,24000,21000,22000],
    'Duration':['30day','40days','35days','40days','60days','50days','55days'],
    'Discount':[1000,2300,1500,1200,2500,2100,2000]
               }
df = pd.DataFrame(technologies)
print(df)
#Pandas Shuffle DataFrame Rows
# shuffle the DataFrame rows & return all rows
df1 = df.sample(frac = 1)
print(df1)
##############################
# Create a new Index starting from zero
df1 = df.sample(frac = 1).reset_index()
print(df1)
############################
# Drop shuffle Index
df1 = df.sample(frac = 1).reset_index(drop=True)
print(df1)
#####################
import pandas as pd
technologies = {
    'Courses':["Spark","PySpark","Python","pandas"],
    'Fee' :[20000,25000,22000,30000],
    'Duration':['30days','40days','35days','50days'],
              }
index_labels=['r1','r2','r3','r4']
df1 = pd.DataFrame(technologies,index=index_labels)

technologies2 = {
    'Courses':["Spark","Java","Python","Go"],
    'Discount':[2000,2300,1200,2000]
              }
index_labels2=['r1','r6','r3','r5']
df2 = pd.DataFrame(technologies2,index=index_labels2)

# pandas join 
df3=df1.join(df2, lsuffix="_left", rsuffix="_right")
print(df3)
#############################
# pandas Inner join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='inner')
print(df3)
##########################
 #pandas Right join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='right')
print(df3)
##########################
#Pandas Merge DataFrames
import pandas as pd
technologies = {
    'Courses':["Spark","PySpark","Python","pandas"],
    'Fee' :[20000,25000,22000,30000],
    'Duration':['30days','40days','35days','50days'],
              }
index_labels=['r1','r2','r3','r4']
df1 = pd.DataFrame(technologies,index=index_labels)

technologies2 = {
    'Courses':["Spark","Java","Python","Go"],
    'Discount':[2000,2300,1200,2000]
              }
index_labels2=['r1','r6','r3','r5']
df2 = pd.DataFrame(technologies2,index=index_labels2)

# Using pandas.merge()
df3= pd.merge(df1,df2)

# Using DataFrame.merge()
df3=df1.merge(df2)
#################################
#Use pandas.concat() to Concat Two DataFrames
import pandas as pd
df = pd.DataFrame({'Courses': ["Spark","PySpark","Python","pandas"],
                    'Fee' : [20000,25000,22000,24000]})

df1 = pd.DataFrame({'Courses': ["Pandas","Hadoop","Hyperion","Java"],
                    'Fee': [25000,25200,24500,24900]})

# Using pandas.concat() to concat two DataFrames
data = [df, df1]
df2 = pd.concat(data)
df2
################################
#Concatenate Multiple DataFrames Using pandas.concat()
import pandas as pd
df = pd.DataFrame({'Courses': ["Spark", "PySpark", "Python", "Pandas"],
                    'Fee' : ['20000', '25000', '22000', '24000']}) 
  
df1 = pd.DataFrame({'Courses': ["Unix", "Hadoop", "Hyperion", "Java"],
                    'Fee': ['25000', '25200', '24500', '24900']})
  
df2 = pd.DataFrame({'Duration':['30day','40days','35days','60days','55days'],
                    'Discount':[1000,2300,2500,2000,3000]})
  
# Appending multiple DataFrame
df3 = pd.concat([df, df1, df2])
print(df3)
##############################
# Write DataFrame to CSV File with Default params.
df3.to_csv("c:/10-python/courses.csv")
#read CSV
# Import pandas
import pandas as pd

# Read CSV file into DataFrame
df = pd.read_csv('courses.csv')
print(df)
############################
# Write DataFrame to Excel file
df.to_excel('c:/10-python/Courses.xlsx')
########################
import pandas as pd
# Read Excel file
df = pd.read_excel('c:/10-python/Courses.xlsx')
print(df)
#########################

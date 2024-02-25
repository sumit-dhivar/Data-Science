# -*- coding: utf-8 -*-
"""
Created on Wed May 31 13:21:46 2023

@author: sumit
"""
#--------------------Assignment on emp_data ---------------------------------
import pandas as pd

df=pd.read_csv("emp_data.csv")
df

df.dtypes

# Convert dtypes 
df=df.convert_dtypes()
df.dtypes

df.columns

df=df.rename(columns={'Salary_hike': 'Hike', 'Churn_out_rate': 'Rate'},inplace = True)
df

# Apply as type
df=df.astype(str)
df.dtypes 
df = df.astype({"Churn_out_rate": float})
print(df.dtypes)

df = df.astype(str).dtypes
df

cols=[ 'Salary_hike','Churn_out_rate']
df[cols] = df[cols].astype('string')
df.dtypes

#Ignores error
df = df.astype({"Churn_out_rate": str},errors='ignore')
df.dtypes

# Generates error
df = df.astype({"Salary_hike": str},errors='raise')

#DataFrame properties
df.shape

df.size

df.columns

df.columns.values

df.index

#Accessing one column contents
df["Salary_hike"]

##Accessing two columns contents
df[["Salary_hike","Churn_out_rate"]]

#select certain rows and assign it to another dataframe
df2=df[6:]
df2

# Accessing certain cell from column
df["Churn_out_rate"][4]



# Describe DataFrame for all numberic columns
df.describe()


#  Rename column 

df2 = df.rename({"Salary_hike": 'SH', 'Churn_out_rate': 'Rate'}, axis='columns')
df2
df2 = df.rename(columns={"Salary_hike": 'SH', 'Churn_out_rate': 'Rate'})

df=df.rename(columns={"Salary_hike": 'SH', 'Churn_out_rate': 'Rate'},inplace=True)
df

df=pd.read_csv("emp_data.csv")
df

# Drop DataFrame rows
df1=df.drop(df.index[1])

df1=df.drop(df.index[1])
df1

df1=df.drop(df.index[[1,3]])
df1

# Delete Rows by Index Range
df1=df.drop(df.index[23:])
df1

# Drop column by names
df1=df.drop(['Salary_hike'],axis=1)
df1

# Labels
df1=df.drop(labels=['Salary_hike'],axis=1)
df1

# columns
df1=df.drop(columns=['Salary_hike'],axis=1)
df1

# Drop Column by index
df.drop(df.columns[1],axis=1)
df

# Drop from Df 
# inplace used for main ope on df
df.drop(df.columns[[2]],axis=1,inplace=True)
df

# Drop two or more columns

lisCol = ["Salary_hike","Hike"]
df2=df.drop(lisCol, axis = 1)
print(df2)

# Drop two or columns by index

df2=df.drop(df.columns[[0,1]],axis=1)

# Drop multiple column
lisCol = ['Salary_hike',"Churn_out_rate"]
df2=df.drop(lisCol, axis = 1)
print(df2)

#Remove columns From DataFrame inplace
df.drop(df.columns[1], axis = 1, inplace=True)
df


df2=df.iloc[:, 0:2]
df2

df2=df.iloc[0:2, :]
df2

#The second slice [:] indicates that all columns are required.

#Slicing Specific Rows and Columns using iloc attribute
df3=df.iloc[1:2, 1:3]
df3
#Another example
df3=df.iloc[:, 1:3]
df3
#The second operator [1:3] yields columns 1 and 3 only.
# Select Rows by Integer Index
df2 = df.iloc[2]
df2


df2 = df.iloc[[2,3,6]]  
df2
df2 = df.iloc[1:5] 
df2
df2 = df.iloc[:1]
df2    
df2 = df.iloc[:3]    

df2 = df.iloc[-3:]
   
df2 = df.iloc[::2]   

# Select Rows by Index Labels
df2 = df.loc[1]   
df2     
df2 = df.loc[[2,3,6]]  
df2  

df2 = df.loc[1:5]    
df2

df2 = df.loc[1:5]
df2 = df.loc[1:5:2]   
df2
##########################################

#Select Rows by Index using Pandas iloc[]
#Select Row by Integer Index

print(df.iloc[2])

# Select Rows by Index List
print(df.iloc[[2,3,6]])


# Select Rows by Integer Index Range
print(df.iloc[1:5])

# Select First Row by Index
print(df.iloc[:1])


# Select First 3 Rows
print(df.iloc[:3])


# Select Last Row by Index
print(df.iloc[-1:])


print(df.iloc[-3:])


df=pd.read_csv("emp_data.csv")
df


df2=df['Salary_hike']
df2

## select multiple columns
df2 = df[['Salary_hike',"Churn_out_rate"]]
df2

# Using loc[] to take column slices
#loc[] syntax to slice columns
#df.loc[:,start:stop:step]
## Selecte multiple columns

df2 = df.loc[:,["Salary_hike","Churn_out_rate"]]
df2

# Select Random columns 

df2 = df.loc[:, ["Salary_hike","Churn_out_rate"]]
df2

# Select columns between two columns 
df2 = df.loc[:,"Salary_hike","Churn_out_rate"]
df2

## Select columns by range
df2 = df.loc[:,"Salary_hike":] 
df2

# Select columns by range 

df2 = df.loc[:,:"Salary_hike"]  

## Select every alternate column
df2 = df.loc[:,::2]
df2       
   


############################

# not equals condition
df2=df.query("Salary_hike !=  1600 , 1706" )
df2

##############################

#Pandas Add Column to DataFrame


#Pandas Add Column to DataFrame
# Add new column to the DataFrame
df2=df.drop(df.index[5:])
df2

Add = ['ABC road', 'BCD Chock', 'Ghansham road', 'Kopargaon', 'Ayodhya']
df2 = df2.assign(Address=Add)
print(df2)


############################

#Append Column to Existing Pandas DataFrame
# Add New column to the existing DataFrame

df2["Address"] = Add
print(df)


#############################

df.columns
print(df.columns)


print(df.columns)

#Quick Examples of Get the Number of Rows in DataFrame
rows_count = len(df.index)
rows_count
rows_count = len(df.axes[0])
rows_count

##################################

row_count = df.shape[0]  # Returns number of rows
col_count = df.shape[1]  # Returns number of columns
print(row_count)

##################################

#pandas Apply Function to a Column
# Below are quick examples
# Using Dataframe.apply() to apply function add column


# Using apply function single column
def add_4(x):
   return x+4
df["Salary_hike"] = df["Salary_hike"].apply(add_4)
df["Salary_hike"]




##############################

#Apply Lambda Function to Single Column
# Using Dataframe.apply() and lambda function


#Using Numpy function on single Column
# Using Dataframe.apply() & [] operator

import numpy as np
df["Salary_hike"] = df["Salary_hike"].apply(np.square)
print(df)



##############################
#Pandas Get Column Names from DataFrame
import pandas as pd
import numpy as np


df.columns
# Get the list of all column names from headers

column_headers = list(df.columns.values)
print("The Column Header :", column_headers)

#Using list(df) to get the column headers as a list
column_headers = list(df.columns)
column_headers

#Using list(df) to get the list of all Column Names
column_headers = list(df)
column_headers

############################

#Pandas Shuffle DataFrame Rows 

#Pandas Shuffle DataFrame Rows

# shuffle the DataFrame rows & return all rows

df1 = df.sample(frac = 1)
print(df1)

# Create a new Index starting from zero
df1 = df.sample(frac = 1).reset_index()
print(df1)

# Drop shuffle Index
df1 = df.sample(frac = 1).reset_index(drop=True)
print(df1)

#pandas inner join is mostly used join, 
#It is used to join two DataFrames on indexes. 
#When indexes don’t match the rows get dropped from both DataFrames.
##################################################

# pandas join ,by default it will join the table left join
df3=df1.join(df2, lsuffix="_left", rsuffix="_right")
print(df3)

# pandas Inner join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='inner')
print(df3)

 #pandas Left join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='left')
print(df3)
 #pandas Right join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='right')
print(df3)



# Get by Column Index
col_list = df[df.columns[0]].values.tolist()
print(col_list)

#--------------------Assignment on ethnic diversity-------------------------
import pandas as pd

df=pd.read_csv("ethnic diversity.csv")
df

df.dtypes

# Convert dtypes 
df=df.convert_dtypes()
df.dtypes

df.columns

df=df.rename(columns={'Employee_Name': 'emp', 'EmpID': 'id'},inplace = True)
df

# Apply as type
df=df.astype(str)
df.dtypes 
df = df.astype({"EmpID": float})
print(df.dtypes)

df = df.astype(str)
df.dtypes

cols=[ 'Employee_Name','Department']
df[cols] = df[cols].astype('string')
df.dtypes

#Ignores error
df = df.astype({"Department": int},errors='ignore')
df.dtypes

# Generates error
df = df.astype({"Department": int},errors='raise')

#DataFrame properties
df.shape

df.size

df.columns

df.columns.values

df.index

#Accessing one column contents
df['Department']

##Accessing two columns contents
df[['Department','Employee_Name']]

#select certain rows and assign it to another dataframe
df2=df[6:]
df2

# Accessing certain cell from column
df['Employee_Name'][4]



# Describe DataFrame for all numberic columns
df.describe()


#  Rename column 

df2 = df.rename({'Employee_Name': 'EN', 'EmpID': 'Eid'}, axis='columns')
df2
df2 = df.rename(columns={'Employee_Name': 'EN', 'EmpID': 'Eid'})

df=df.rename(columns={'Employee_Name': 'EN', 'EmpID': 'Eid'},inplace=True)
df

df=pd.read_csv("C:/1-python/ethnic diversity.csv")
df

# Drop DataFrame rows
df1=df.drop(df.index[1])

df1=df.drop(df.index[1])
df1

df1=df.drop(df.index[[1,3]])
df1

# Delete Rows by Index Range
df1=df.drop(df.index[23:])
df1

# Drop column by names
df1=df.drop(['Employee_Name'],axis=1)
df1

# Labels
df1=df.drop(labels=['Employee_Name'],axis=1)
df1

# columns
df1=df.drop(columns=['Employee_Name'],axis=1)
df1

# Drop Column by index
df.drop(df.columns[1],axis=1)
df

# Drop from Df 
# inplace used for main ope on df
df.drop(df.columns[[2]],axis=1,inplace=True)
df

# Drop two or more columns

lisCol = ["Employee_Name","EmpID"]
df2=df.drop(lisCol, axis = 1)
print(df2)

# Drop two or columns by index

df2=df.drop(df.columns[[0,1]],axis=1)

# Drop multiple column
lisCol = ["Employee_Name","EmpID"]
df2=df.drop(lisCol, axis = 1)
print(df2)

#Remove columns From DataFrame inplace
df.drop(df.columns[1], axis = 1, inplace=True)
df


df2=df.iloc[:, 0:2]
df2

df2=df.iloc[0:2, :]
df2

#The second slice [:] indicates that all columns are required.

#Slicing Specific Rows and Columns using iloc attribute
df3=df.iloc[1:2, 1:3]
df3
#Another example
df3=df.iloc[:, 1:3]
df3
#The second operator [1:3] yields columns 1 and 3 only.
# Select Rows by Integer Index
df2 = df.iloc[2]
df2


df2 = df.iloc[[2,3,6]]  
df2
df2 = df.iloc[1:5] 
df2
df2 = df.iloc[:1]
df2    
df2 = df.iloc[:3]    

df2 = df.iloc[-3:]
   
df2 = df.iloc[::2]   

# Select Rows by Index Labels
df2 = df.loc[1]   
df2     
df2 = df.loc[[2,3,6]]  
df2  

df2 = df.loc[1:5]    
df2

df2 = df.loc[1:5]
df2 = df.loc[1:5:2]   
df2
##########################################

#Select Rows by Index using Pandas iloc[]
#Select Row by Integer Index

print(df.iloc[2])

# Select Rows by Index List
print(df.iloc[[2,3,6]])


# Select Rows by Integer Index Range
print(df.iloc[1:5])

# Select First Row by Index
print(df.iloc[:1])


# Select First 3 Rows
print(df.iloc[:3])


# Select Last Row by Index
print(df.iloc[-1:])


print(df.iloc[-3:])


df=pd.read_csv("C:/1-python/ethnic diversity.csv")
df


df2=df["Employee_Name"]
df2

## select multiple columns
df2 = df[["Employee_Name","EmpID"]]
df2

# Using loc[] to take column slices
#loc[] syntax to slice columns
#df.loc[:,start:stop:step]
## Selecte multiple columns

df2 = df.loc[:,["Employee_Name","EmpID"]]
df2

# Select Random columns 

df2 = df.loc[:, ["Employee_Name","EmpID"]]
df2

# Select columns between two columns 
df2 = df.loc[:,"Employee_Name","EmpID"]
df2

## Select columns by range
df2 = df.loc[:,"Employee_Name":] 
df2

# Select columns by range 

df2 = df.loc[:,:'Employee_Name']  

## Select every alternate column
df2 = df.loc[:,::2]
df2       
   


############################

# not equals condition
df2=df.query("Employee_Name != 'Brown, Mia'" )
df2

##############################

#Pandas Add Column to DataFrame


#Pandas Add Column to DataFrame
# Add new column to the DataFrame
df2=df.drop(df.index[5:])
df2

Add = ['ABC road', 'BCD Chock', 'Ghansham road', 'Kopargaon', 'Ayodhya']
df2 = df2.assign(Address=Add)
print(df2)


############################

#Append Column to Existing Pandas DataFrame
# Add New column to the existing DataFrame

df2["Address"] = Add
print(df)


#############################

df.columns
print(df.columns)


print(df.columns)

#Quick Examples of Get the Number of Rows in DataFrame
rows_count = len(df.index)
rows_count
rows_count = len(df.axes[0])
rows_count

##################################

row_count = df.shape[0]  # Returns number of rows
col_count = df.shape[1]  # Returns number of columns
print(row_count)

##################################

#pandas Apply Function to a Column
# Below are quick examples
# Using Dataframe.apply() to apply function add column


# Using apply function single column
def add_4(x):
   return x+4
df["EmpID"] = df["EmpID"].apply(add_4)
df["EmpID"]




##############################

#Apply Lambda Function to Single Column
# Using Dataframe.apply() and lambda function


#Using Numpy function on single Column
# Using Dataframe.apply() & [] operator

import numpy as np
df['EmpID'] = df['EmpID'].apply(np.square)
print(df)



##############################
#Pandas Get Column Names from DataFrame
import pandas as pd
import numpy as np


df.columns
# Get the list of all column names from headers

column_headers = list(df.columns.values)
print("The Column Header :", column_headers)

#Using list(df) to get the column headers as a list
column_headers = list(df.columns)
column_headers

#Using list(df) to get the list of all Column Names
column_headers = list(df)
column_headers

############################

#Pandas Shuffle DataFrame Rows 

#Pandas Shuffle DataFrame Rows

# shuffle the DataFrame rows & return all rows

df1 = df.sample(frac = 1)
print(df1)

# Create a new Index starting from zero
df1 = df.sample(frac = 1).reset_index()
print(df1)

# Drop shuffle Index
df1 = df.sample(frac = 1).reset_index(drop=True)
print(df1)

#pandas inner join is mostly used join, 
#It is used to join two DataFrames on indexes. 
#When indexes don’t match the rows get dropped from both DataFrames.
##################################################

# pandas join ,by default it will join the table left join
df3=df1.join(df2, lsuffix="_left", rsuffix="_right")
print(df3)

# pandas Inner join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='inner')
print(df3)

 #pandas Left join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='left')
print(df3)
 #pandas Right join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='right')
print(df3)



# Get by Column Index
col_list = df[df.columns[0]].values.tolist()
print(col_list)

#---------------------Assignment on Diabetes-------------------------------
import pandas as pd

df=pd.read_csv("Diabetes.csv")
df

df.dtypes

# Convert dtypes 
df=df.convert_dtypes()
df.dtypes

df.columns

df=df.rename(columns={'Triceps skin fold thickness': 'Triceps thickness', '2-Hour serum insulin': 'serum insulin'},inplace = True)
df

# Apply as type
df=df.astype(str)
df.dtypes 
df = df.astype({"Diastolic blood pressure": float})
print(df.dtypes)

df = df.astype(str)
df.dtypes

cols=[ 'Diastolic blood pressure','Diabetes pedigree function']
df[cols] = df[cols].astype('string')
df.dtypes

#Ignores error
df = df.astype({"Class variable": int},errors='ignore')
df.dtypes

# Generates error
df = df.astype({"Class variable": int},errors='raise')

#DataFrame properties
df.shape

df.size

df.columns

df.columns.values

df.index

#Accessing one column contents
df['Number of times pregnant']

##Accessing two columns contents
df[['Number of times pregnant','Plasma glucose concentration']]

#select certain rows and assign it to another dataframe
df2=df[6:]
df2

# Accessing certain cell from column
df['Plasma glucose concentration'][4]



# Describe DataFrame for all numberic columns
df.describe()


#  Rename column 

df2 = df.rename({'Diastolic blood pressure': 'Diastolic_BP', 'Plasma glucose concentration': 'Plasma_GC'}, axis='columns')
df2
df2 = df.rename(columns={'Diastolic blood pressure': 'Diastolic_BP', 'Plasma glucose concentration': 'Plasma_GC'})

df=df.rename(columns={'Diastolic blood pressure': 'Diastolic_BP', 'Plasma glucose concentration': 'Plasma_GC'},inplace=True)
df

df=pd.read_csv("Diabetes.csv")
df

# Drop DataFrame rows
df1=df.drop(df.index[1])

df1=df.drop(df.index[1])
df1

df1=df.drop(df.index[[1,3]])
df1

# Delete Rows by Index Range
df1=df.drop(df.index[23:])
df1

# Drop column by names
df1=df.drop(['Diabetes pedigree function'],axis=1)
df1

# Labels
df1=df.drop(labels=['Diabetes pedigree function'],axis=1)
df1

# columns
df1=df.drop(columns=['Diabetes pedigree function'],axis=1)
df1

# Drop Column by index
df.drop(df.columns[1],axis=1)
df

# Drop from Df 
# inplace used for main ope on df
df.drop(df.columns[[2]],axis=1,inplace=True)
df

# Drop two or more columns

lisCol = ['Diabetes pedigree function','Body mass index']
df2=df.drop(lisCol, axis = 1)
print(df2)

# Drop two or columns by index

df2=df.drop(df.columns[[0,1]],axis=1)

# Drop multiple column
lisCol = ['Diabetes pedigree function','Body mass index']
df2=df.drop(lisCol, axis = 1)
print(df2)

#Remove columns From DataFrame inplace
df.drop(df.columns[1], axis = 1, inplace=True)
df


df2=df.iloc[:, 0:2]
df2

df2=df.iloc[0:2, :]
df2

#The second slice [:] indicates that all columns are required.

#Slicing Specific Rows and Columns using iloc attribute
df3=df.iloc[1:2, 1:3]
df3
#Another example
df3=df.iloc[:, 1:3]
df3
#The second operator [1:3] yields columns 1 and 3 only.
# Select Rows by Integer Index
df2 = df.iloc[2]
df2


df2 = df.iloc[[2,3,6]]  
df2
df2 = df.iloc[1:5] 
df2
df2 = df.iloc[:1]
df2    
df2 = df.iloc[:3]    

df2 = df.iloc[-3:]
   
df2 = df.iloc[::2]   

# Select Rows by Index Labels
df2 = df.loc[1]   
df2     
df2 = df.loc[[2,3,6]]  
df2  

df2 = df.loc[1:5]    
df2

df2 = df.loc[1:5]
df2 = df.loc[1:5:2]   
df2
##########################################

#Select Rows by Index using Pandas iloc[]
#Select Row by Integer Index

print(df.iloc[2])

# Select Rows by Index List
print(df.iloc[[2,3,6]])


# Select Rows by Integer Index Range
print(df.iloc[1:5])

# Select First Row by Index
print(df.iloc[:1])


# Select First 3 Rows
print(df.iloc[:3])


# Select Last Row by Index
print(df.iloc[-1:])


print(df.iloc[-3:])


df=pd.read_csv("Diabetes.csv")
df


df2=df['Diabetes pedigree function']
df2

## select multiple columns
df2 = df[['Diabetes pedigree function','Body mass index']]
df2

# Using loc[] to take column slices
#loc[] syntax to slice columns
#df.loc[:,start:stop:step]
## Selecte multiple columns

df2 = df.loc[:,['Diabetes pedigree function','Body mass index']]
df2

# Select Random columns 

df2 = df.loc[:, ['Diabetes pedigree function','Body mass index']]
df2

# Select columns between two columns 
df2 = df.loc[:,'Diabetes pedigree function','Body mass index']
df2

## Select columns by range
df2 = df.loc[:,"Class variable":] 
df2

# Select columns by range 

df2 = df.loc[:,:'Class variable']  

## Select every alternate column
df2 = df.loc[:,::2]
df2       
   


############################

# not equals condition
df2=df.query("Class variable != 'NO'" )
df2

##############################

#Pandas Add Column to DataFrame


#Pandas Add Column to DataFrame
# Add new column to the DataFrame
df2=df.drop(df.index[5:])
df2

Add = ['ABC road', 'BCD Chock', 'Ghansham road', 'Kopargaon', 'Ayodhya']
df2 = df2.assign(Address=Add)
print(df2)


############################

#Append Column to Existing Pandas DataFrame
# Add New column to the existing DataFrame

df2["Address"] = Add
print(df)


#############################

df.columns
print(df.columns)


print(df.columns)

#Quick Examples of Get the Number of Rows in DataFrame
rows_count = len(df.index)
rows_count
rows_count = len(df.axes[0])
rows_count

##################################

row_count = df.shape[0]  # Returns number of rows
col_count = df.shape[1]  # Returns number of columns
print(row_count)

##################################

#pandas Apply Function to a Column
# Below are quick examples
# Using Dataframe.apply() to apply function add column


# Using apply function single column
def add_4(x):
   return x+4
df["Class variable"] = df["Class variable"].apply(add_4)
df["Class variable"]




##############################

#Apply Lambda Function to Single Column
# Using Dataframe.apply() and lambda function


#Using Numpy function on single Column
# Using Dataframe.apply() & [] operator

import numpy as np
df['Class variable'] = df['Class variable'].apply(np.square)
print(df)



##############################
#Pandas Get Column Names from DataFrame
import pandas as pd
import numpy as np


df.columns
# Get the list of all column names from headers

column_headers = list(df.columns.values)
print("The Column Header :", column_headers)

#Using list(df) to get the column headers as a list
column_headers = list(df.columns)
column_headers

#Using list(df) to get the list of all Column Names
column_headers = list(df)
column_headers

############################

#Pandas Shuffle DataFrame Rows 

#Pandas Shuffle DataFrame Rows

# shuffle the DataFrame rows & return all rows

df1 = df.sample(frac = 1)
print(df1)

# Create a new Index starting from zero
df1 = df.sample(frac = 1).reset_index()
print(df1)

# Drop shuffle Index
df1 = df.sample(frac = 1).reset_index(drop=True)
print(df1)

#pandas inner join is mostly used join, 
#It is used to join two DataFrames on indexes. 
#When indexes don’t match the rows get dropped from both DataFrames.
##################################################

# pandas join ,by default it will join the table left join
df3=df1.join(df2, lsuffix="_left", rsuffix="_right")
print(df3)

# pandas Inner join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='inner')
print(df3)

 #pandas Left join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='left')
print(df3)
 #pandas Right join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='right')
print(df3)

# Get by Column Index
col_list = df[df.columns[0]].values.tolist()
print(col_list)


#-----------------Assignment on Computer_Data-----------------------------
import pandas as pd

df=pd.read_csv("Computer_Data.csv")
df

df.dtypes

# Convert dtypes 
df=df.convert_dtypes()
df.dtypes

df.columns

df=df.rename(columns={'Unnamed: 0': 'Sr_No', 'price': 'Rate'},inplace = True)
df

# Apply as type
df=df.astype(str)
df.dtypes 
df = df.astype({"hd": float})
print(df.dtypes)

df = df.astype(str)
df.dtypes

cols=[ 'speed', 'hd']
df[cols] = df[cols].astype('string')
df.dtypes

#Ignores error
df = df.astype({'screen': int},errors='ignore')
df.dtypes

# Generates error
df = df.astype({'screen': int},errors='raise')

#DataFrame properties
df.shape

df.size

df.columns

df.columns.values

df.index

#Accessing one column contents
df['multi']

##Accessing two columns contents
df[['ram', 'screen']]

#select certain rows and assign it to another dataframe
df2=df[6:]
df2

# Accessing certain cell from column
df['price'][4]



# Describe DataFrame for all numberic columns
df.describe()


#  Rename column 

df2 = df.rename({'hd':'HD', 'ram':'RAM'}, axis='columns')
df2
df2 = df.rename(columns={'hd':'HD', 'ram':'RAM'})

df=df.rename(columns={'hd':'HD', 'ram':'RAM'},inplace=True)
df

df=pd.read_csv("Computer_Data.csv")
df

# Drop DataFrame rows
df1=df.drop(df.index[1])

df1=df.drop(df.index[1])
df1

df1=df.drop(df.index[[1,3]])
df1

# Delete Rows by Index Range
df1=df.drop(df.index[23:])
df1

# Drop column by names
df1=df.drop(['screen'],axis=1)
df1

# Labels
df1=df.drop(labels=['screen'],axis=1)
df1

# columns
df1=df.drop(columns=['premium'],axis=1)
df1

# Drop Column by index
df.drop(df.columns[1],axis=1)
df

# Drop from Df 
# inplace used for main ope on df
df.drop(df.columns[[2]],axis=1,inplace=True)
df

# Drop two or more columns

lisCol = ['ads', 'trend']
df2=df.drop(lisCol, axis = 1)
print(df2)

# Drop two or columns by index

df2=df.drop(df.columns[[0,1]],axis=1)

# Drop multiple column
lisCol = ['ads', 'trend']
df2=df.drop(lisCol, axis = 1)
print(df2)

#Remove columns From DataFrame inplace
df.drop(df.columns[1], axis = 1, inplace=True)
df


df2=df.iloc[:, 0:2]
df2

df2=df.iloc[0:2, :]
df2

#The second slice [:] indicates that all columns are required.

#Slicing Specific Rows and Columns using iloc attribute
df3=df.iloc[1:2, 1:3]
df3
#Another example
df3=df.iloc[:, 1:3]
df3
#The second operator [1:3] yields columns 1 and 3 only.
# Select Rows by Integer Index
df2 = df.iloc[2]
df2


df2 = df.iloc[[2,3,6]]  
df2
df2 = df.iloc[1:5] 
df2
df2 = df.iloc[:1]
df2    
df2 = df.iloc[:3]    

df2 = df.iloc[-3:]
   
df2 = df.iloc[::2]   

# Select Rows by Index Labels
df2 = df.loc[1]   
df2     
df2 = df.loc[[2,3,6]]  
df2  

df2 = df.loc[1:5]    
df2

df2 = df.loc[1:5]
df2 = df.loc[1:5:2]   
df2
##########################################

#Select Rows by Index using Pandas iloc[]
#Select Row by Integer Index

print(df.iloc[2])

# Select Rows by Index List
print(df.iloc[[2,3,6]])


# Select Rows by Integer Index Range
print(df.iloc[1:5])

# Select First Row by Index
print(df.iloc[:1])


# Select First 3 Rows
print(df.iloc[:3])


# Select Last Row by Index
print(df.iloc[-1:])


print(df.iloc[-3:])


df=pd.read_csv("Computer_Data.csv")
df


df2=df['premium']
df2

## select multiple columns
df2 = df[['premium', 'ads']]
df2

# Using loc[] to take column slices
#loc[] syntax to slice columns
#df.loc[:,start:stop:step]
## Selecte multiple columns

df2 = df.loc[:,['premium', 'ads']]
df2

# Select Random columns 

df2 = df.loc[:, ['premium', 'ads']]
df2

# Select columns between two columns 
df2 = df.loc[:,'premium', 'ads']
df2

## Select columns by range
df2 = df.loc[:,'trend':] 
df2

# Select columns by range 

df2 = df.loc[:,:'trend']  

## Select every alternate column
df2 = df.loc[:,::2]
df2       
   


############################

# not equals condition
df2=df.query("cd != 'no'" )
df2

##############################

#Pandas Add Column to DataFrame


#Pandas Add Column to DataFrame
# Add new column to the DataFrame
df2=df.drop(df.index[5:])
df2

Add = ['ABC road', 'BCD Chock', 'Ghansham road', 'Kopargaon', 'Ayodhya']
df2 = df2.assign(Address=Add)
print(df2)


############################

#Append Column to Existing Pandas DataFrame
# Add New column to the existing DataFrame

df2["Address"] = Add
print(df)


#############################

df.columns
print(df.columns)


print(df.columns)

#Quick Examples of Get the Number of Rows in DataFrame
rows_count = len(df.index)
rows_count
rows_count = len(df.axes[0])
rows_count

##################################

row_count = df.shape[0]  # Returns number of rows
col_count = df.shape[1]  # Returns number of columns
print(row_count)

##################################

#pandas Apply Function to a Column
# Below are quick examples
# Using Dataframe.apply() to apply function add column


# Using apply function single column
def add_4(x):
   return x+4
df['ads'] = df['ads'].apply(add_4)
df['ads']




##############################

#Apply Lambda Function to Single Column
# Using Dataframe.apply() and lambda function


#Using Numpy function on single Column
# Using Dataframe.apply() & [] operator

import numpy as np
df['ads'] = df['ads'].apply(np.square)
print(df)



##############################
#Pandas Get Column Names from DataFrame
import pandas as pd
import numpy as np


df.columns
# Get the list of all column names from headers

column_headers = list(df.columns.values)
print("The Column Header :", column_headers)

#Using list(df) to get the column headers as a list
column_headers = list(df.columns)
column_headers

#Using list(df) to get the list of all Column Names
column_headers = list(df)
column_headers

############################

#Pandas Shuffle DataFrame Rows 

#Pandas Shuffle DataFrame Rows

# shuffle the DataFrame rows & return all rows

df1 = df.sample(frac = 1)
print(df1)

# Create a new Index starting from zero
df1 = df.sample(frac = 1).reset_index()
print(df1)

# Drop shuffle Index
df1 = df.sample(frac = 1).reset_index(drop=True)
print(df1)

#pandas inner join is mostly used join, 
#It is used to join two DataFrames on indexes. 
#When indexes don’t match the rows get dropped from both DataFrames.
##################################################

# pandas join ,by default it will join the table left join
df3=df1.join(df2, lsuffix="_left", rsuffix="_right")
print(df3)

# pandas Inner join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='inner')
print(df3)

 #pandas Left join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='left')
print(df3)

 #pandas Right join DataFrames
df3=df1.join(df2, lsuffix="_left", rsuffix="_right", how='right')
print(df3)

# Get by Column Index
col_list = df[df.columns[0]].values.tolist()
print(col_list)




















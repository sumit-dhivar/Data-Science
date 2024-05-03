# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:36:32 2024

@author: sumit

Title:- Agglomerative Clustering
DataSet:- Telco_customer_churn
"""
"""
Data Dictionary:- 
Feature	              Relevant	                     Description
Customer ID	            No	             Unique identifier for each customer

Count	                No	                   Count of occurrences 

Quarter	                No	                     Quarter of the year

Referred a Friend	    Yes	       Indicates whether the customer was referred by a friend (Yes/No)

Number of Referrals	    Yes	        The count of referrals made by the customer

Tenure in Months	    Yes	       Duration of the customer's subscription in months

Offer	               Yes	           Type of offer provided to the customer

Phone Service	       Yes	       Indicates whether the customer subscribes to phone service (Yes/No)

Avg Monthly Long       Yes	         Average monthly long-distance charges incurred by the customer
Distance Charges	

Multiple Lines	       Yes	           Indicates whether the customer has multiple phone lines (Yes/No)

Internet Service	   Yes	        Type of internet service subscribed by the customer

Internet Type	       Yes	                   Type of internet connection

Avg Monthly GB         Yes	   Average monthly gigabytes downloaded by the customer
Download	

Online Security	       Yes	      Indicates whether the customer has online security service (Yes/No)

Online Backup	       Yes	      Indicates whether the customer has online backup service (Yes/No)

Device Protection      Yes	      Indicates whether the customer has a device protection plan (Yes/No)
 Plan 

Premium Tech Support	Yes	       Indicates whether the customer has premium technical support (Yes/No)

Streaming TV	       Yes	        Indicates whether the customer subscribes to streaming TV (Yes/No)

Streaming Movies	   Yes	        Indicates whether the customer subscribes to streaming movies (Yes/No)

Streaming Music	       Yes	         Indicates whether the customer subscribes to streaming music (Yes/No)

Unlimited Data	      Yes	        Indicates whether the customer has unlimited data plan (Yes/No)

Contract	          Yes	        Type of contract subscribed by the customer (e.g., month-to-month, one year, two years)

Paperless Billing	 Yes	        Indicates whether the customer opts for paperless billing (Yes/No)

Payment Method	     Yes	       Method of payment used by the customer

Monthly Charge	     Yes	       Monthly charge incurred by the customer

Total Charges	    Yes	           Total charges incurred by the customer

Total Refunds	   Yes	           Total refunds provided to the customer

Total Extra Data    Yes             Total extra data charges incurred by the customer
Charges		

Total Long          Yes	               Total long-distance charges incurred by the customer
Distance Charges	

Total Revenue	    Yes	              Total revenue generated from the customer (sum of charges minus refunds plus extra data charges)

"""

"""
:)Buisness Problem:-
=>Buisness Objective:- 
1) The primary goal is likely to reduce the churn rate of telecom customers.
2) he objective could be to segment customers into distinct groups based on their behavior,
 preferences, and characteristics. 
3) provide insights into the factors that influence customers' decisions to churn.

=>Buisness Constraints:-
The availability and quality of data could be a constraint.
If the dataset is incomplete, contains errors, or lacks certain key variables,
 it may limit the accuracy and effectiveness of the clustering analysis and 
insights derived from it.
"""



import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df = pd.read_excel('Telco_customer_churn.xlsx')

df.shape
#(7043, 30)
#So we can see that we are having almost 30 features in our dataset 
#From these most of the features will be irrelevant to our clustering, 
#hence if this features will not be correlated then we will drop the 
#unneccesary features to reduce the computational complexity of the clustering algo

df.columns
# ['Customer ID', 'Count', 'Quarter', 'Referred a Friend',
#        'Number of Referrals', 'Tenure in Months', 'Offer', 'Phone Service',
#        'Avg Monthly Long Distance Charges', 'Multiple Lines',
#        'Internet Service', 'Internet Type', 'Avg Monthly GB Download',
#        'Online Security', 'Online Backup', 'Device Protection Plan',
#        'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
#        'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing',
#        'Payment Method', 'Monthly Charge', 'Total Charges', 'Total Refunds',
#        'Total Extra Data Charges', 'Total Long Distance Charges',
#        'Total Revenue'] 
#These are the features of the data set 

#Before proceeding for further analysis we will drop those features
#which are not correlated, relevant and rename the features according to standard
df.drop({'Customer ID'},axis=1,inplace=True)
#As this customer id is irrelevant to us

df.rename({'Referred a Friend':'referred_a_friend',
           'Number of Referrals':'number_of_referals',
           'Tenure in Months':'tenure_in_months',
           'Phone Service':'phone_service',
           'Avg Monthly Long Distance Charges':'avg_monthly_long_distance_charges',
           'Multiple Lines':'multiple_lines',
           'Internet Service':'internet_service',
           'Internet Type':'internet_type',
           'Avg Monthly GB Download':'avg_monthly_gb_download',
           'Online Security':'online_security',
           'Online Backup':'online_backup',
           'Device Protection Plan':'device_protection_plan',
           'Premium Tech Support':'premium_tech_support',
           'Streaming TV':'streaming_tv',
           'Streaming Movies':'streaming_movies',
           'Streaming Music':'streaming_movies',
           'Unlimited Data':'unlimited_data',
           'Paperless Billing':'paperless_billing',           
           'Payment Method':'payment_method',
           'Monthly Charge':'monthly_charge',
           'Total Charges':'total_charges',
           'Total Refunds':'total_refunds',
           'Total Extra Data Charges':'total_extra_data_charges',
           'Total Long Distance Charges':'total_long_distance_charges',
           'Total Revenue':'total_revenue'
           },axis=1,inplace=True)

cols = df.columns
cols
# ['Count', 'number_of_referals', 'tenure_in_months',
#        'avg_monthly_long_distance_charges', 'avg_monthly_gb_download',
#        'monthly_charge', 'total_charges', 'total_refunds',
#        'total_extra_data_charges', 'total_long_distance_charges',
#        'total_revenue', 'Quarter', 'referred_a_friend', 'Offer',
#        'phone_service', 'multiple_lines', 'internet_service', 'internet_type',
#        'online_security', 'online_backup', 'device_protection_plan',
#        'premium_tech_support', 'streaming_tv', 'streaming_movies',
#        'streaming_movies', 'unlimited_data', 'Contract', 'paperless_billing',
#        'payment_method']
#Renamed Columns
#Now we can see that we are having two features with same name
#i.e streaming_movies so we will rename one of the feature 

list(cols).index("streaming_movies")
#So the first occurence of streaming_movies is at the position 23
#So we will rename this feature 
df.columns.values[17] = 'streaming_movies1'

df = df.iloc[:,[0,1,2,3,11,12,13,14,15,16,17,4,5,6,7,8,9,10,18,19,20,21,22,23,24,25,26,27,28]]
#rearranged the features for simplicity

df_categorical = df.iloc[:,11:]
df_categorical.columns


for i in df_categorical.columns:
    print(f"For the feature {i}: - ")
    print(df[i].value_counts())
    print("------------------------")

#So from this we can see that the Quarter feature has only one type of category
#hence we will drop this feature 

df.drop({'Quarter'},axis=1,inplace=True)






















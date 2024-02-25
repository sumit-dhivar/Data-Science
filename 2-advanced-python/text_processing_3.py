# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 08:23:52 2023

@author: sumit
"""
#Pattern to find mobile number in text.
import re 
text1='My mobile number is 8983525299'
text2='My altername mobile number is 9952528389'
text3='My International mobile number is (124)-456-74589'

pat = '\d{10}'
re.findall(pat,text1)

pat='(?:.....)-\d{3}-\d{5}'
re.findall(pat,text3)

pat = '\(\d{3}\)-\d{3}-\d{5}'
re.findall(pat,text3)

#Pattern to find EmailID in text
import re 
text1 = 'My email is sumitdhivar86103@gmail.com'
text2 = 'My college mail ID is sumitdhivarcomp@sanjivanicoe.org.in'

pat3 = '[a-z0-9]*@[a-z]*\.[a-z]*'
re.findall(pat3,text1) 

pat3 = '[a-z0-9]*@[a-z]*\.[a-z]*\.[a-z]*'
re.findall(pat3,text2)


#Order number
text4 = 'Hi my order #496724 , is not received'
text5 = 'Hi I am having problem with order number 496724.'
text6 = 'Hi my order number 496724 is having issue'

pat4 = '\#\d{6}'
re.findall(pat4,text4)

pat5 = '\d{6}\.'
re.findall(pat5,text5)

pat6 = '\d{6}'
re.findall(pat6,text6)

pat = 'order[^\d]*(\d*)'
x=re.findall(pat,text5)
x

re.findall(pat,text4)

re.findall(pat,text6)
#==================================================================
def pattern_match(pat,text):
    matches = re.findall(pat,text)
    if matches:
        return matches 
pattern_match('order[^\d]*(\d*)', text4)

t="""Born	Elon Reeve Musk
June 28, 1971 (age 51)
Pretoria, Transvaal, South Africa
Education	University of Pennsylvania (BA, BS)
Title	
Founder, CEO and chief engineer of SpaceX
CEO and product architect of Tesla, Inc.
Owner, CTO and chairman of Twitter
President of the Musk Foundation
Founder of the Boring Company, X Corp. and X.AI
Co-founder of Neuralink, OpenAI, Zip2 and X.com (part of PayPal)
Spouses	
Justine Wilson
​
​(m. 2000; div. 2008)​
Talulah Riley
​
​(m. 2010; div. 2012)​
​
​(m. 2013; div. 2016)​
Partner	Grimes (2018–2021)[1]
Children	10[a][3]
Parents	
Errol Musk (father)
Maye Musk (mother)
Family	Musk family"""

pat='Born(\s*[a-zA-Z]*\s*[a-zA-Z]*\s*[a-zA-Z]*)'
re.findall(pat,t)

pattern_match('Born(.*)', t)
#['\tElon Reeve Musk']

x = pattern_match('Born(.*)', t)
x
x[0].strip()
#'Elon Reeve Musk'

#Born.*\n(.*)

pattern_match(r'Born.*\n(.*)\(age',t)
pattern_match(r'\age.*\n(.*)', t)

#=========================8/06/23================================= 
def extract_personal_info(text):
    age = pattern_match('age (\d+)', text)
    full_name = pattern_match('Born(.*)\n', text)
    birth_date = pattern_match('Born.*\n(.*)\(age', text)
    birth_place = pattern_match('\(age.*\n(.*)', text)
    return {
        'age':age,
        'name':full_name,
        'birth_date':birth_date,
        'birth_place':birth_place
        }
extract_personal_info(t)

#===================================================================
ambani="""Born	Mukesh Dhirubhai Ambani
19 April 1957 (age 66)
Aden, Colony of Aden
(present-day Yemen)[1][2]
Nationality	Indian
Alma mater	
St. Xavier's College, Mumbai
Institute of Chemical Technology (B.E.)
Occupation(s)	Chairman and MD, Reliance Industries
Spouse	Nita Ambani ​(m. 1985)​[3]
Children	3
Parent	
Dhirubhai Ambani (father)
Relatives	Anil Ambani (brother)
Tina Ambani (sister-in-law)"""

def extract_personal_info(text):
    age = pattern_match('age (\d+)', text)
    full_name = pattern_match('Born(.*)\n', text)
    birth_date = pattern_match('Born.*\n(.*)\(age', text)
    birth_place = pattern_match('\(age.*\n(.*)', text)
    return {
        'age':age,
        'name':full_name,
        'birth_date':birth_date,
        'birth_place':birth_place
        }
extract_personal_info(ambani)

#---------------------------------------------------------------
#Pattern to find the twitter account name
import re 
text1 = '''
Follow our leader Elon musk on twitter here: https://twitter.com/elonmusk, more information 
on Tesla's products can be found at https://www.tesla.com/. Also here are leading influencers 
for tesla related news,
https://twitter.com/teslarati
https://twitter.com/dummy_tesla
https://twitter.com/dummy_2_tesla
'''
pat = 'https://twitter.com/([a-zA-Z0-9_]+)'
pattern_match(pat, text1)

#Extract Concentration of Risks Types. It will 
t1 = """Concentration of Risk: Credit Risk
Financial instruments that potentially subject us to a concentration of credit risk consist of cash, cash equivalents, marketable securities,
restricted cash, accounts receivable, convertible note hedges, and interest rate swaps. Our cash balances are primarily invested in money market funds
or on deposit at high 
Concentration of Risk: Supply Risk credit quality financial institutions in the U.S. These deposits are typically in excess of insured limits. As of September 30, 2021
and December 31,"""
pattern = 'Concentration of Risk: ([^\n]* Risk)'

pattern_match(pattern, t1)



text ='''
Tesla's gross cost of operating lease vehicles in FY2021 Q1
was $4.85 billion..
BMW's gross cost of operating vehicles in FY2021 S1 was $8.
billion.'''

pattern ='FY(\d{4} (?:Q[1-4]|S[1-2]))'
matches = re.findall(pattern, text)
matches

pattern = 'FY(\d{4} (?:Q[1-4]|S[1-4]))'
pattern = 'FY(\d{4} (?:Q[1-4]|S[1-4]) was \$(\d)+)'
pattern_match(pattern, text)

#match this ?:

text='''
Elon musk's phone number is 9991116666, call him if you have any questions on dodgecoin. Tesla's revenue is 40 billion
Tesla's CFO number (999)-333-7777
'''
pattern = '\(\d{3}\)-\d{3}-\d{4}|\d{10}'
pattern_match(pattern, text)


text = '''
Note 1 - Overview
Tesla, Inc. (“Tesla”, the “Company”, “we”, “us” or “our”) was incorporated in the State of Delaware on July 1, 2003. We design, develop, manufacture and sell high-performance fully electric vehicles and design, manufacture, install and sell solar energy generation and energy storage
products. Our Chief Executive Officer, as the chief operating decision maker (“CODM”), organizes our company, manages resource allocations and measures performance among two operating and reportable segments: (i) automotive and (ii) energy generation and storage.
Beginning in the first quarter of 2021, there has been a trend in many parts of the world of increasing availability and administration of vaccines
against COVID-19, as well as an easing of restrictions on social, business, travel and government activities and functions. On the other hand, infection
rates and regulations continue to fluctuate in various regions and there are ongoing global impacts resulting from the pandemic, including challenges
and increases in costs for logistics and supply chains, such as increased port congestion, intermittent supplier delays and a shortfall of semiconductor
supply. We have also previously been affected by temporary manufacturing closures, employment and compensation adjustments and impediments to
administrative activities supporting our product deliveries and deployments.
Note 2 - Summary of Significant Accounting Policies
Unaudited Interim Financial Statements
The consolidated balance sheet as of September 30, 2021, the consolidated statements of operations, the consolidated statements of
comprehensive income, the consolidated statements of redeemable noncontrolling interests and equity for the three and nine months ended September
30, 2021 and 2020 and the consolidated statements of cash flows for the nine months ended September 30, 2021 and 2020, as well as other information
disclosed in the accompanying notes, are unaudited. The consolidated balance sheet as of December 31, 2020 was derived from the audited
consolidated financial statements as of that date. The interim consolidated financial statements and the accompanying notes should be read in
conjunction with the annual consolidated financial statements and the accompanying notes contained in our Annual Report on Form 10-K for the year
ended December 31, 2020.
'''
pattern = 'Note \d -([^\n]*)'
pattern_match(pattern, text)


text = '''
The gross cost of operating lease vehicles in FY2021 Q1 was $4.85 billion.
In previous quarter i.e. FY2020 Q4 it was $3 billion. 
'''
pattern = 'FY\d{4} Q[1-4]'
pattern_match(pattern, text)


text = '''
The gross cost of operating lease vehicles in FY2021 Q1 was $4.85 billion.
In previous quarter i.e. FY2020 Q4 it was $3 billion. 
'''
# pattern = 'FY\d{4} Q[1-4]'
# pattern_match(pattern, text)
matches = re.findall(pattern,text,flags = re.IGNORECASE)
matches



















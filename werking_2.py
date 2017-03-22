#read in libraries

#Tectia 

from __future__ import division
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from pandas import ExcelWriter
import numpy as np
import gzip
import seaborn as sns

from scipy import stats
from math import exp

import re
import string


#read in area code and state mapping excel sheet
areaCode_state = pd.read_excel('area_codes_by_state.xls') #read in from Desktop Copy


#Either read in FPO.APO table or Apo_Consolidation.sas output in excel #or select columns in SAS from beginning.
#read in email lds data from John


#emailz= pd.read_excel('APO4.xlsx')    
#emailz= pd.read_csv('/data/1/namcards/fraud/MySASDataset.csv') #mixed dtypes.
df_ww = pd.read_csv('/data/1/namcards/fraud/WWDataset.csv')

emails = df_ww[['RESP_STAT_DT','APPL_ID_NUM', 'ACQ_TID', 'ASIGN_CREDT_LIMIT', 'PRIM_APPL_NM', 'PRIM_ADDR_DESC','PRIM_CITY_CD', 'PRIM_STATE_CD','PRIM_HM_PHONE_NUM','CELL_PH_NBR', 'EMAIL_ADDR_DESC', 'AUTH_APPL_NM','PRIM_DOB_DT', 'PORTF_ID']]

#emails = emailz[['RESP_STAT_DT','APPL_ID_NUM', 'ACQ_TID', 'ASIGN_CREDT_LIMIT', 'PRIM_APPL_NM', 'PRIM_ADDR_DESC',
#'PRIM_CITY_CD', 'PRIM_STATE_CD',
#'PRIM_HM_PHONE_NUM','CELL_PH_NBR', 'EMAIL_ADDR_DESC', 'AUTH_APPL_NM', 
#'PRIM_DOB_DT', 'PORTF_ID']]

#error: http://stackoverflow.com/questions/28682562/pandas-read-csv-converting-mixed-types-columns-as-string

#missing values
emails.isnull().sum()
#verify below is necessary later. 

#>>> emails[emails['EMAIL_ADDR_DESC'].isnull()].APPL_ID_NUM
#>>> df_ww[df_ww['EMAIL_ADDR_DESC'].isnull()].APPL_DECSN_DESC.value_counts()/df_ww.APPL_DECSN_DESC.value_counts()
#Approve    0.096581
#201702219180408


emails['PRIM_HM_PHONE_NUM'].fillna('-', inplace=True)
emails['CELL_PH_NBR'].fillna('-', inplace=True)
emails['AUTH_APPL_NM'].fillna('-', inplace=True)
emails['PRIM_APPL_NM'].fillna('-:-', inplace=True)
emails['PRIM_DOB_DT'].fillna('-,-', inplace=True)

#missing emails
emails['EMAIL_ADDR_DESC'].fillna('-@-.-', inplace=True)

emails.isnull().sum()
#---------------------------------------------
emx = emails['PRIM_ADDR_DESC'].apply(str)
#-----------------------------------------------------
localpart_numbers = []
for i in emails['EMAIL_ADDR_DESC']: 
    
    lp = i.rsplit('@', 1)[0] #localpart str
    lpt_n = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lp)
    lpt_n = ', '.join(map(str, lpt_n))
    
    if len(lpt_n) == 0:
        localpart_numbers.append(0) #'-'
    else:
        localpart_numbers.append(lpt_n)

addr_numbercount = []
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in emx: 
    if i != '_':
        addr_numbercount.append(count(i, string.digits))
    else:
        addr_numbercount.append(0)
        
addr_numbers = []
for i in emx:
    lpt_n = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", i)
    lpt_n = ', '.join(map(str, lpt_n))
    #print(type(lpt_n)) #list to str
    if len(lpt_n) == 0:
        addr_numbers.append('na') 
    else:
        addr_numbers.append(lpt_n)

        
addr_letters = []
for i in emx: 
    #print(i, type(i))
    if i != '_':
        #print(i)
        addr_letters.append(re.findall('[a-zA-Z]+', i))
    else:
        addr_letters.append(0)


#addr_letters 
addr_letter_count = []
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in emx: #emails['PRIM_ADDR_DESC']:#Vxlsx['Address1']:
    if i != '_':
        addr_letter_count.append(count(i, string.ascii_letters))
    else:
        lettercount.append(0)

#addr_letter_count
#number of address words
addr_wordLength = []
for i in addr_letters:
    addr_wordLength.append(len(i))
    #print(len(i))

#compare localpart_numbers with addr_numbers
#create dataframe 
#convert strings back to integer
comp_df = pd.DataFrame({'localpart_numbers':localpart_numbers, 'addr_numbers':addr_numbers})
comp_df['match'] = comp_df['localpart_numbers'] == comp_df['addr_numbers']

#
comp_df[comp_df['match'] == True]

cmp_df = pd.DataFrame({'localpart_numbers':localpart_numbers, 'addr_numbers':addr_numbers})
cmp_df['match'] = comp_df['localpart_numbers'] == comp_df['addr_numbers']
cmp_df[cmp_df['match'] == True]

DOB_yr = []
for i in emails.PRIM_DOB_DT:
    try:
        DOB_yr.append(i.year)
        #print(i.year)
    except AttributeError:
        DOB_yr.append(0)

print(len(DOB_yr))

phoneNumb = []
for em in emails['PRIM_HM_PHONE_NUM']:
    if em != '-':
        phoneNumb.append(int(em))
    else:
        phoneNumb.append(em)

emails['PRIM_HM_PHONE_NUM'] = phoneNumb

cellNumb = []
for em in emails['PRIM_HM_PHONE_NUM']:
    if em != '-':
        cellNumb.append(int(em))
    else:
        cellNumb.append(em)

emails['CELL_PH_NBR'] = cellNumb


firstName = []
lastName = []
for i in emails['PRIM_APPL_NM']:
    fn = re.split(r'[:| ]', i)[0]
    fn = re.sub(' [a-z]*', '', fn)
    ln = re.split(r':', i)[1]
    firstName.append(fn)
    lastName.append(ln)


Domain = []
DomainName = []
DomainEnd = []

for i in emails['EMAIL_ADDR_DESC']:
    if re.findall('@', i):
        dm = i.rsplit('@', 1)[1]
        dmn = dm.rsplit('.', 1)[0]
        Domain.append(dm)
        DomainName.append(dmn)
        
        if len(i.rsplit('.', 1)) < 2: #splitting by period. if no domainEnd exists then
            DomainEnd.append('-')
        else:
            new = dm.rsplit('.', 1)
            if len(new) < 2:
                print('only one')
                DomainEnd.append(new)
            else:
                dme = dm.rsplit('.', 1)[1]
                DomainEnd.append(dme)

    else: #is just a word and there's no email
        Domain.append('-')
        DomainName.append('-')
        DomainEnd.append('-')

localPart = []
for i in emails['EMAIL_ADDR_DESC']:#.dropna():
    #print(len(i))  #str, float
    lp = i.rsplit('@', 1)[0]
    localPart.append(lp)
    
emailLength = []
for i in emails['EMAIL_ADDR_DESC']:#.dropna():
    #print(i)
    #print(len(i))  #str, float 
    emailLength.append(len(i))

new_df = pd.DataFrame({'FName':firstName, 'LName':lastName,'Phone': phoneNumb, 'Cell': cellNumb,'Email_Total_Length': emailLength, 'LocalPart': localPart,'Domain': Domain, 'Domain_Name':DomainName,'Domain_End':DomainEnd})

LP_length = []
for i in new_df['LocalPart'].dropna():
    #print(len(i))  #str, float 
    LP_length.append(len(i))  

stopcount=[]
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in new_df.LocalPart:
    if i != '_':
        #print(i, count(i,string.punctuation))
        stopcount.append(count(i, string.punctuation))
    else:
        #print(i)
        stopcount.append(i)

numbercount=[]
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in new_df.LocalPart:
    if i != '_':
        numbercount.append(count(i, string.digits))
    else:
        numbercount.append(i)

lettercount=[]
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in new_df.LocalPart:
    if i != '_':
        lettercount.append(count(i, string.ascii_letters))
    else:
        lettercount.append(i)
        
fn_Match = []
for x in range(0,len(new_df)): #len of new_df
    imtag = re.search(new_df.FName[x], new_df.LocalPart[x])
    #print(imtag)
    if imtag:
        #print("{}".format(imtag.group(0)))
        fn_Match.append("{}".format(imtag.group(0)))
    else:
        #print("-")
        fn_Match.append("-")
        
ln_Match = []
for x in range(0,len(new_df)):
    imtag = re.search(new_df.LName[x], new_df.LocalPart[x])
    #print(imtag)
    if imtag:
        #print("{}".format(imtag.group(0)))
        ln_Match.append("{}".format(imtag.group(0)))
    else:
        #print("-")
        ln_Match.append("-")


#2.22 adding in APP_ID
new_df = pd.DataFrame({'ACQ_TID': emails['ACQ_TID'], 'App_ID': emails['APPL_ID_NUM'], 'App_Date': emails['RESP_STAT_DT'], 'State': emails['PRIM_STATE_CD'],
'DOB': DOB_yr,'FName':firstName, 'LName':lastName,'Phone': phoneNumb,'Cell': cellNumb,
'Email_Total_Length': emailLength,'LocalPart': localPart, 'LP_length': LP_length,'Domain': Domain,'Domain_Name':DomainName,'Domain_End':DomainEnd,'stops':stopcount,'FN_match':fn_Match,'LN_match':ln_Match,'LP_lettercnt':lettercount, 'LP_numbercnt':numbercount})

#print(new_df.shape)

FirstName_match = []
for i in new_df.FN_match:
    if i == '-':
        FirstName_match.append(int(0))
        #print('0')
    else:
        FirstName_match.append(int(1))
        #int('1')

LastName_match = []
for i in new_df.LN_match:
    if i == '-':
        LastName_match.append(int(0))
        #print('0')
    else:
        LastName_match.append(int(1))


new_df = pd.DataFrame({'ACQ_TID': emails['ACQ_TID'], 'App_ID': emails['APPL_ID_NUM'],'Address': emails['PRIM_ADDR_DESC'],'App_Date': emails['RESP_STAT_DT'], 'State': emails['PRIM_STATE_CD'], 'DOB': DOB_yr,'FName':firstName, 'LName':lastName,'Phone': phoneNumb, 'Cell': cellNumb, 
'Email_Total_Length': emailLength, 'LocalPart': localPart, 'LP_length': LP_length,'Domain': Domain, 'Domain_Name':DomainName,'Domain_End':DomainEnd,'stops':stopcount,'FN_match':FirstName_match, 'LN_match':LastName_match, 'LP_lettercnt':lettercount, 'LP_numbercnt':numbercount})

new_df['numbers'] = new_df.LP_numbercnt/new_df.LP_length
#LP_numbercnt/LP_length
new_df['stopz'] = new_df.stops/new_df.LP_length
#stops/LP_length
new_df['letters'] = new_df.LP_lettercnt/new_df.LP_length
#LP_lettercnt/LP_length
new_df['FN_and_LN_match'] = new_df['FN_match'] + new_df['LN_match']


new_df['FN_LN_match'] = np.where(new_df['FN_and_LN_match']==2, 1, 0)

del new_df['FN_and_LN_match']
#check this


cell_area_code = []
for i in new_df['Cell']:
    if i == '-':
        cell_area_code.append(i)
    else:
        ac = str(i)[:3]
        ac = int(ac)
        cell_area_code.append(ac)

area_code = []
for i in new_df['Phone']:
    if i == '-':
        area_code.append(i)
    else:
        ac = str(i)[:3]
        ac = int(ac)
        area_code.append(ac)
    
#read in areacode and state excel sheet
#join based on area code to see corresponding state

area_code = pd.DataFrame({'Area code': area_code})                
#area_code.head(4)

cell_area_code = pd.DataFrame({'Area code': cell_area_code})
#cell_area_code.head(4)

cell_area_code_state = cell_area_code.merge(areaCode_state, on='Area code', how='left')
area_code_state =area_code.merge(areaCode_state, on='Area code', how='left')
new_df['area_code_state'] = area_code_state['State code']
new_df['cell_area_code_state'] = cell_area_code_state['State code']

new_df['area_code_state'] = new_df['area_code_state'].fillna('-')
new_df['cell_area_code_state'] = new_df['area_code_state'].fillna('-')

newer_df=new_df[new_df['area_code_state'] != '-']

newer_df['que'] = newer_df['State'] == newer_df['area_code_state'] #T/F
newer_df['cell_que'] = newer_df['State'] == newer_df['cell_area_code_state']
newer_df[newer_df['que'] == False].head(24)
newer_df[newer_df['que'] == False].shape #1423 total no matches with state and areacode

newest_df = newer_df.que

newest_df = pd.DataFrame({'que': newest_df})

emails_df = new_df.merge(newest_df, left_index=True, right_index=True, how='left')
emails_df['que'].fillna('-', inplace=True)
emails_df = emails_df.rename(index=str, columns={"que": "Phone_State_Match"})

Age = []
for i in emails_df['DOB']:
    if i <= 1928:
        Age.append('88-98')
    elif 1928 < i <= 1938:
        Age.append('78-88')
    elif 1938 < i <= 1948:
        #print(i)
        Age.append('68-78')
    elif 1948 < i <= 1958:
        Age.append('58-68')
    elif 1958 < i <= 1968:
        #print(i)
        Age.append('48-58')
    elif 1968 < i <= 1978:
        Age.append('38-48')
    elif 1978 < i <= 1988:
        #print(i)
        Age.append('28-38')
    elif 1988 < i <= 1998:
        #print(i)
        Age.append('18-28')
    elif 1998 < i <= 2016:
        #print(i)
        Age.append('0-18')
    else:
        Age.append('?')

emails_df['Age_bucket'] = Age
emails_df['Age'] = 2016 - emails_df['DOB']

flse = newer_df[newer_df.que == False]
#flse[flse['xFraud_App'] == 1]

emails_df.columns

emails_data = emails_df

emails_data = emails_data.drop(['App_Date', 'Cell', 'cell_area_code_state'], axis=1, inplace=True)

#del emals_df['FN_and_LN_match']

comp_df['addr_numbercount'] = addr_numbercount
comp_df['addr_wordLength'] = addr_wordLength
comp_df['addr_letter_count'] = addr_letter_count
indx = comp_df.index.tolist()
comp_df['index'] = indx
comp_df['match'].value_counts() #613
comp_df.head(3)

print(comp_df.shape)


print(emails_df.shape) #(90362, 25)
emails_df['index'] = emails_df.index.tolist()
emails_df['index']= emails_df['index'].apply(str)
comp_df['index']= comp_df['index'].apply(str)
result = pd.merge(emails_df, comp_df, on='index')


emails_result_df = result


#df = emails_result_df.append(emals_result_df)

df = emails_result_df

del df['Phone']

test = df['Domain'].str.upper()

test = test.reset_index()
test['Domain'].apply(lambda x: x.lower()).head(3)


df['Domain'] = df['Domain'].apply(lambda x: x.upper())
df['Domain_End'] = df['Domain_End'].apply(lambda x: x.upper())
df['Domain_Name'] = df['Domain_Name'].apply(lambda x: x.upper())
df['FName'] = df['FName'].apply(lambda x: x.upper())
df['LName'] = df['LName'].apply(lambda x: x.upper())
df['LocalPart'] = df['LocalPart'].apply(lambda x: x.upper())


#
lookup = df.groupby(["Domain_Name"]).size().reset_index(name="total_cnt")

#what to do with lookup and freqtable? 
#2/21
#a_c =lookup.merge(freqtable, on='Domain_Name', how='left')
#print(a_c.shape)


domain_length = []
for i in df['Domain_Name'].dropna():
    #print(len(i))  #str, float 
    domain_length.append(len(i))  

stopcount_domain=[]
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in df.Domain_Name:
    if i != '_':
        #print(i, count(i,string.punctuation))
        stopcount_domain.append(count(i, string.punctuation))
    else:
        #print(i)
        stopcount_domain.append(i)

numbercount_domain=[]
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in df.Domain_Name:
    if i != '_':
        numbercount_domain.append(count(i, string.digits))
    else:
        numbercount_domain.append(i)

lettercount_domain=[]
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in df.Domain_Name:
    if i != '_':
        lettercount_domain.append(count(i, string.ascii_letters))
    else:
        lettercount_domain.append(i)

print(len(lettercount_domain))

vowel_count = []
vowels = set("AEIOU")
for domain in df.Domain_Name: #letter vs. word...GMAIL
    count = 0
    vowlz=[]
    for letter in domain:
        if letter in vowels:
            #vowlz=[]
            vowlz.append(letter)
    vowel_count.append(len(vowlz))


print(len(vowel_count))


#export emails
#emails_result_df.to_csv("emails.csv") #== df_ ?? 

writer = ExcelWriter('emails.xlsx')
emails_result_df.to_excel(writer,'Sheet1')
writer.save()

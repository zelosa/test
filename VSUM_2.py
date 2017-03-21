
# coding: utf-8

# In[37]:

#read in libraries
#get_ipython().magic('pwd')
#get_ipython().magic('cd C:\\Users\\ww32293\\Desktop')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import re
import numpy as np
import pandas as pd
#get_ipython().magic('matplotlib inline')
import matplotlib.dates as mdates
#from __future__ import division
from scipy import stats


# VSUM

# In[38]:

#VSUM Daily Reports
#adds to fraud app samples
#verify against


# In[39]:

#read in #VSUM Daily Reports
Vxlsx= pd.read_excel('VSUM_111316_0111_.xlsx') #, header = 1)
Vxlsx.columns


# In[40]:

#replace missing values with '-''

Vxlsx['State'].fillna('-', inplace=True)
Vxlsx['Email Address'].fillna('_@_._', inplace=True)
Vxlsx['Primary Name'].fillna('-,-', inplace=True)
k = Vxlsx['Email Address'].replace(['na'], ['_@_.com'])
Vxlsx['Email Address']= k
#phone
Vxlsx['Home Phone'].fillna('-', inplace=True)
Vxlsx['Business Phone'].fillna('-', inplace=True)
#DOB
Vxlsx['DOB'].fillna('-,-', inplace=True)
#Address
Vxlsx['Address1'].fillna('-', inplace=True)
#any patterns in missingness?


# In[41]:

#Compare numbers in email local part with numbers in Address
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
#furmanzachary9

localpart_numbers = []
for i in Vxlsx['Email Address']:
    #print(len(i))  #str, float
    lp = i.rsplit('@', 1)[0] #localpart str
    lpt_n = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lp)
    lpt_n = ', '.join(map(str, lpt_n))
    #print(type(lpt_n)) #list to str
    if len(lpt_n) == 0:
        localpart_numbers.append('-')
    else:
        localpart_numbers.append(lpt_n)
        
print(len(localpart_numbers))

#number of numbers in address#
addr_numbercount = []
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in Vxlsx['Address1']:
#     #rint(type(i))
    if i != '_':
        addr_numbercount.append(count(i, string.digits))
    else:
        addr_numbercount.append(0)
        
addr_numbers = []
for i in Vxlsx['Address1']:
    #print(len(i))  #str, float
    #lp = i.rsplit('@', 1)[0] #localpart str
    lpt_n = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", i)
    lpt_n = ', '.join(map(str, lpt_n))
    #print(type(lpt_n)) #list to str
    if len(lpt_n) == 0:
        addr_numbers.append('na')
    else:
        addr_numbers.append(lpt_n)
        
        
#len(addr_numbercount) #8009
#addr_numbercount

#Address street name and last part like CT/RD/LN 
#collect words into element in list

addr_letters = []
for i in Vxlsx['Address1']:
    if i != '_':
        addr_letters.append(re.findall('[a-zA-Z]+', i))
    else:
        addr_letters.append(0)

#addr_letters 

addr_letter_count = []
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in Vxlsx['Address1']:
    if i != '_':
        addr_letter_count.append(count(i, string.ascii_letters))
#     else:
#         lettercount.append(0)
    
#number of address words
addr_wordLength = []
for i in addr_letters:
    addr_wordLength.append(len(i))
    #print(len(i))

#len(addr_wordLength)
#addr_wordLength

#compare localpart_numbers with addr_numbers
#create temp dataframe 
comp_df = pd.DataFrame({'localpart_numbers':localpart_numbers, 'addr_numbers':addr_numbers})
comp_df['match'] = comp_df['localpart_numbers'] == comp_df['addr_numbers']

#29
#comp_df[comp_df['match'] == True].index


# In[42]:

comp_df['match'].value_counts()


# In[43]:

comp_df['addr_numbercount'] = addr_numbercount
comp_df['addr_wordLength'] = addr_wordLength
comp_df['addr_letter_count'] = addr_letter_count
comp_df['Address'] = Vxlsx['Address1']
indx = comp_df.index.tolist()
comp_df['index'] = indx

#1/19 index check


# In[44]:

#comp_df.shape #(8009, 7)
comp_df.head(3)


# In[45]:

firstname = []
lastname = []

for i in Vxlsx['Primary Name']:
    #print(i)
    #if find ',' in i: 
    #fn = i.rsplit('; |, |\*',1)#[1].lower() #IndexError: list index out of range
    fn = re.split(r'[ ,\s]\s*', i)[1].lower()
    ln = re.split(r'[ ,\s]\s*', i)[0].lower()
    firstname.append(fn)
    lastname.append(ln)

domain = []
domainName = []
domainEnd = []
for i in Vxlsx['Email Address']: #k = V['Email Address']
    #print(i)  #str, float
    #dm = re.split(r'@*',i)[1]
    dm = i.rsplit('@', 1)[1]
    dmn = dm.rsplit('.', 1)[0]
    dme = dm.rsplit('.', 1)[1]
    domain.append(dm)
    domainName.append(dmn)
    domainEnd.append(dme)

localpart = []
for i in Vxlsx['Email Address'].dropna():
    #print(len(i))  #str, float
    lp = i.rsplit('@', 1)[0]
    localpart.append(lp)
    
length = []
for i in Vxlsx['Email Address'].dropna():
    #print(len(i))  #str, float 
    length.append(len(i))


# In[46]:

from collections import Counter
counts = Counter(domainEnd)
print(counts)

#function
#Counter, takes top and makes the feature itself? for now the given dataset. nonstationary data later. 

#feature importance plot will at least show whats important to the model
#model will predict if given fraud_app is fradulent or not
    #don't care about interpretability?
#logistic regression, decision tree, PCA...

#dummifying categorical variables vs. one-hot-encoding and assigning numbers/ranks



# In[47]:

# from collections import Counter
# counts = Counter(domainName)
# print(counts)

#spam/ham


# In[48]:

n_df = pd.DataFrame({'FName':firstname, 'LName':lastname, 'Email_Total_Length': length, 
        'LocalPart': localpart, 'Domain_Name':domainName,'Domain_End':domainEnd})

LP_length = []
for i in n_df['LocalPart'].dropna():
    #print(len(i))  #str, float 
    LP_length.append(len(i))  

n_df = pd.DataFrame({'FName':firstname, 'LName':lastname, 'Email_Total_Length': length, 
        'LocalPart': localpart, 'Domain_Name':domainName,'Domain_End':domainEnd, 'LP_length': LP_length,})

stopcount=[]
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in n_df.LocalPart:
    if i != '_':
        #print(i, count(i,string.punctuation))
        stopcount.append(count(i, string.punctuation))
    else:
        #print(i)
        stopcount.append(0)

numbercount=[]
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in n_df.LocalPart:
    if i != '_':
        numbercount.append(count(i, string.digits))
    else:
        numbercount.append(0)

lettercount=[]
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in n_df.LocalPart:
    if i != '_':
        lettercount.append(count(i, string.ascii_letters))
    else:
        lettercount.append(0)
        
fnMatch = []
for x in range(0, len(n_df)):
    imtag = re.search(n_df.FName[x], n_df.LocalPart[x])
    #print(imtag)
    if imtag:
        #print("{}".format(imtag.group(0)))
        fnMatch.append("{}".format(imtag.group(0)))
    else:
        #print("-")
        fnMatch.append("-")
lnMatch = []
for x in range(0,len(n_df)):
    imtag = re.search(n_df.LName[x], n_df.LocalPart[x])
    #print(imtag)
    if imtag:
        #print("{}".format(imtag.group(0)))
        lnMatch.append("{}".format(imtag.group(0)))
    else:
        #print("-")
        lnMatch.append("-")
        
n_df = pd.DataFrame({'FName':firstname, 'LName':lastname, 'Email_Total_Length': length, 
        'LocalPart': localpart, 'LP_length': LP_length,'Domain':domain, 'Domain_Name':domainName,'Domain_End':domainEnd, 'stops':stopcount, 
                    'FN_match':fnMatch, 'LN_match':lnMatch, 'LP_lettercnt':lettercount, 'LP_numbercnt':numbercount})
#n_df.head(5)


# In[49]:

FirstName_match = []
for i in n_df.FN_match:
    if i == '-':
        FirstName_match.append(int(0))
        #print('0')
    else:
        FirstName_match.append(int(1))
        #int('1')

LastName_match = []
for i in n_df.LN_match:
    if i == '-':
        LastName_match.append(int(0))
        #print('0')
    else:
        LastName_match.append(int(1))
        #int('1')

n_df = pd.DataFrame({'FName':firstname, 'LName':lastname, 'Email_Total_Length': length, 
        'LocalPart': localpart, 'LP_length': LP_length,'Domain':domain, 'Domain_Name':domainName,'Domain_End':domainEnd, 'stops':stopcount, 
                    'FN_match':FirstName_match, 'LN_match':LastName_match, 'LP_lettercnt':lettercount, 'LP_numbercnt':numbercount})
#n_df.head(5)  


# In[50]:

phoneNumber = []
for em in Vxlsx['Home Phone']:
    if em != '-':
        phoneNumber.append(int(em))
    else:
        phoneNumber.append(em)
        
bizphoneNumber = []
for em in Vxlsx['Business Phone']:
    if em != '-':
        bizphoneNumber.append(int(em))
    else:
        bizphoneNumber.append(em)


# In[51]:

#if homephone = null, use business phone instead?
#buisness phone is null
#homephone and business phone are the same 


# In[52]:

DOB_year = []
for i in Vxlsx['DOB']:
    try:
        DOB_year.append(i.year)
        #print(i.year)
    except AttributeError:
        DOB_year.append(0) #'-'
        #print('-')
n_df['DOB'] = DOB_year


# In[53]:

n_df['numbers'] = n_df.LP_numbercnt/n_df.LP_length
n_df['stopz'] = n_df.stops/n_df.LP_length
n_df['letters'] = n_df.LP_lettercnt/n_df.LP_length


# In[54]:

n_df['FN_and_LN_match'] = n_df['FN_match'] + n_df['LN_match']
n_df['FN_and_LN_match'].head(4)
n_df['FN_LN_match'] = np.where(n_df['FN_and_LN_match']==2, 1, 0)


# In[55]:

#AreaCode
a_c = []
for i in phoneNumber:
    if i == '-':
        a_c.append(i)
    else:
        ac = str(i)[:3]
        ac = int(ac)
        a_c.append(ac)
    
#read in areacode and state excel sheet
#join based on area code to see corresponding state

a_c = pd.DataFrame({'Area code': a_c})                
a_c.head(4)


# In[56]:

#read in area code and state mapping excel sheet
areaCode_state = pd.read_excel('area_codes_by_state.xls')


# In[57]:

a_c_state =a_c.merge(areaCode_state, on='Area code', how='left') #area_code_state


# In[58]:

#n_df.columns
#Add DOB, #Domain, Phone, State, #xFraud_App, numbers, stopz, letters, Phone_State_Match, Age_bucket, Age


# In[59]:

#n_df.head(3)


# In[60]:

n_df['DOB'] = DOB_year
n_df['State'] = Vxlsx['State']
n_df['area_code_state'] = a_c_state['State code']
n_df['area_code_state'] = n_df['area_code_state'].fillna('-')

#xFraud_App == 1 for all
#address_mm DNE in VSUM daily reports
#no App_Date here

n_df['xFraud_App'] = 1

n_df.head(5)


# In[61]:

#n_df.columns
#Phone_State_Match
#FN_LN_match
#Age_bucket
#Age


# In[62]:

#checking unknowns Phone State match
#n_df.FN_and_LN_match.isnull
n_df.shape
sum(n_df.FN_and_LN_match.isnull())


# In[63]:

newr_df=n_df[n_df['area_code_state'] != '-']
newr_df.shape #3885


# In[64]:

#newr_df.head(3)


# In[65]:

#n_df['area_code_state'] = n_df['area_code_state'].fillna('-')
newr_df['que'] = newr_df['State'] == newr_df['area_code_state'] #T/F
newr_df[newr_df['que'] == False].shape #1284
newr_df.shape #3885 ... 33% of frauds have state not matching area code on their phones


# In[66]:

nwest_df = newr_df.que
nwest_df.index


# In[67]:

#nwest_df.head(4)


# In[68]:

nwest_df = pd.DataFrame({'que': nwest_df})


# In[69]:

#nwest_df.head(10)


# In[70]:

#n_df.head(5)


# In[71]:

#emals_df.columns


# In[72]:

#merging n_df and nwest_df
emals_df = n_df.merge(nwest_df, left_index=True, right_index=True, how='left')

emals_df['que'].fillna('-', inplace=True)
emals_df = emals_df.rename(index=str, columns={"que": "Phone_State_Match"})

emals_df.head(3)


# In[73]:

#emals_df.shape #(8009, 23)


# In[74]:

#Age bucketing 

Agee = []
for i in emals_df['DOB']:
    if i <= 1928:
        Agee.append('88-98')
    elif 1928 < i <= 1938:
        Agee.append('78-88')
    elif 1938 < i <= 1948:
        #print(i)
        Agee.append('68-78')
    elif 1948 < i <= 1958:
        Agee.append('58-68')
    elif 1958 < i <= 1968:
        #print(i)
        Agee.append('48-58')
    elif 1968 < i <= 1978:
        Agee.append('38-48')
    elif 1978 < i <= 1988:
        #print(i)
        Agee.append('28-38')
    elif 1988 < i <= 1998:
        #print(i)
        Agee.append('18-28')
    elif 1998 < i <= 2016:
        #print(i)
        Agee.append('0-18')
    else:
        Agee.append('?')
        
        
len(Agee)


# In[75]:

emals_df['Age_bucket'] = Agee
emals_df['Age'] = 2016 - emals_df['DOB']
emals_df.head(3)


# In[76]:

emals_df.Age_bucket.value_counts()


# In[1]:

emals_df.columns #25


# In[78]:

emals_df.head(3)


# In[79]:

emals_df.shape #append on comp_df


# In[80]:

comp_df.shape
comp_df.head(3)


# In[81]:

emals_df['index'] = emals_df.index.tolist()
emals_df['index']= emals_df['index'].apply(str)
comp_df['index']= comp_df['index'].apply(str)
result = pd.merge(emals_df, comp_df, on='index')
result.shape
result.head(2)


# In[82]:

len(emals_df.index.unique())


# In[83]:

#len(emails_df.index.unique())


# In[84]:

#len(emails_df) #index restarts count once emals is appended to it


# In[85]:

emals_result_df = result
#emals_result_df.shape #(8009, 31)


# In[86]:

#result.columns


# In[87]:

#result 8009,33 for emals_df
#other dataset with same columns


#add on together
#dummify domains
#start model


# In[88]:

#1/19


# In[89]:

#Domain popularity 
#dummify categorical variables


# In[ ]:




# In[ ]:




# lds report - from John

# In[90]:

#read in area code and state mapping excel sheet
areaCode_state = pd.read_excel('area_codes_by_state.xls')


# In[91]:

#read in email lds data from John
emailz= pd.read_excel('email_lds2.xlsx')
#emailz.shape #(24098, 25)


# In[92]:

emailz_2= pd.read_excel('email_lds_sample2.xlsx')
#emailz_2.shape #(66264, 25)


# In[93]:

emailz.columns


# In[94]:

#1/18/2017
#zip codes? 


# In[95]:

len(emailz.PRIM_ADDR_DESC)


# In[96]:

#need to setup training and test sets - shuffle later


# In[97]:

#subset raw data 
emails = emailz[['app_date', 'ASIGN_CREDT_LIMIT', 'PRIM_APPL_NM', 'PRIM_ADDR_DESC','PRIM_CITY_CD', 'PRIM_STATE_CD',
                 'PRIM_HM_PHONE_NUM','CELL_PH_NBR', 'EMAIL_ADDR_DESC', 'AUTH_APPL_NM', 
                 'PRIM_DOB_DT', 'PORTF_ID', 'fraud_app', 'addr_mm']]
emails.shape


# In[98]:

emails_2 = emailz_2[['app_date', 'ASIGN_CREDT_LIMIT', 'PRIM_APPL_NM', 'PRIM_ADDR_DESC','PRIM_CITY_CD', 'PRIM_STATE_CD',
                 'PRIM_HM_PHONE_NUM','CELL_PH_NBR', 'EMAIL_ADDR_DESC', 'AUTH_APPL_NM', 
                 'PRIM_DOB_DT', 'PORTF_ID', 'fraud_app', 'addr_mm']]
#fraud_app_ind
emails_2.shape
#why isn't it reading anything more?


# In[99]:

#append datasets
emails = emails.append(emails_2, ignore_index=True)

emails.shape #(90362, 14)


# In[100]:

emails[90361:90362]


# In[101]:

#emails_2['EMAIL_ADDR_DESC'] #8289


# In[102]:

#missing values
emails.isnull().sum() #9566 missing


# In[103]:

emails_2.isnull().sum() #8289 missing emails


# In[104]:

emails['PRIM_HM_PHONE_NUM'].fillna('-', inplace=True)
emails['CELL_PH_NBR'].fillna('-', inplace=True)
emails['AUTH_APPL_NM'].fillna('-', inplace=True)
emails['PRIM_APPL_NM'].fillna('-,-', inplace=True)
emails['PRIM_DOB_DT'].fillna('-,-', inplace=True)

#missing emails
emails['EMAIL_ADDR_DESC'].fillna('-@-.-', inplace=True)


# In[105]:

#emails[emails['EMAIL_ADDR_DESC'] == 'LYNNBOWLING@GMAIL.COM']


# In[106]:

emails.isnull().sum()


# In[107]:

#len(emails['EMAIL_ADDR_DESC']) #90362


# In[108]:

emails['PRIM_ADDR_DESC'].isnull().sum()


# In[109]:

emails['EMAIL_ADDR_DESC'][0:4]


# In[110]:

emx = emails['PRIM_ADDR_DESC'].apply(str)


# In[111]:

#1/19?
localpart_numbers = []
for i in emails['EMAIL_ADDR_DESC']: #Vxlsx['Email Address']:
    #print(len(i))  #str, float
    lp = i.rsplit('@', 1)[0] #localpart str
    lpt_n = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", lp)
    lpt_n = ', '.join(map(str, lpt_n))
    #print(type(lpt_n)) #list to str
    if len(lpt_n) == 0:
        localpart_numbers.append(0) #'-'
    else:
        localpart_numbers.append(lpt_n)
        
print(len(localpart_numbers))

#localpart_numbers
#number of numbers in address#
addr_numbercount = []
import string
count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
for i in emx: #emails['PRIM_ADDR_DESC']: #Vxlsx['Address1']:
#     #rint(type(i))
    if i != '_':
        addr_numbercount.append(count(i, string.digits))
    else:
        addr_numbercount.append(0)
        
addr_numbers = []
for i in emx:#emails['PRIM_ADDR_DESC']:#Vxlsx['Address1']:  #make integer into str? 
    lpt_n = re.findall("[-+]?\d+[\.]?\d*[eE]?[-+]?\d*", i)
    lpt_n = ', '.join(map(str, lpt_n))
    #print(type(lpt_n)) #list to str
    if len(lpt_n) == 0:
        addr_numbers.append('na') 
    else:
        addr_numbers.append(lpt_n)

        
#len(addr_numbercount)
#addr_numbercount
#Address street name and last part like CT/RD/LN 
#collect words into element in list

addr_letters = []
for i in emx: #emails['PRIM_ADDR_DESC']:#Vxlsx['Address1']:
    #print(i, type(i))
    if i != '_':
        #print(i)
        addr_letters.append(re.findall('[a-zA-Z]+', i))
    else:
        addr_letters.append(0)

#fix
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

#29
comp_df[comp_df['match'] == True]

#add as new feature onto final dataframe 
#kept addr_numbers empty instead of str '-' ?

#comp_df.head(30)


# In[112]:

#emails['PRIM_ADDR_DESC'][241:242]
#241    9702 BRAES VALLEY ST
#241    LUISHERNANDEZ9702@GMAIL.COM
#201610183030205


# In[113]:

#emails['EMAIL_ADDR_DESC'][241:242]


# In[114]:

cmp_df = pd.DataFrame({'localpart_numbers':localpart_numbers, 'addr_numbers':addr_numbers})
cmp_df['match'] = comp_df['localpart_numbers'] == comp_df['addr_numbers']
cmp_df[cmp_df['match'] == True]
#wanted to verify against email localpart
#would have to pull index, join against emails['EMAIL_ADDR_DESC']


# In[115]:

comp_df[comp_df['match'] == True].index #613


# In[116]:

#len(addr_numbercount) #90362


# In[117]:

#Extract Year from DOB column
DOB_yr = []
for i in emails.PRIM_DOB_DT:
    try:
        DOB_yr.append(i.year)
        #print(i.year)
    except AttributeError:
        DOB_yr.append(0) #'-'
        #print('-')
len(DOB_yr)

#phone number

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

#cell number


# In[118]:

#firstName, lastname
firstName = []
lastName = []

for i in emails['PRIM_APPL_NM']:
    #remove middle name ?
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
            dme = dm.rsplit('.', 1)[1]
            DomainEnd.append(dme)

    else: #is just a word and there's no email
        Domain.append('-')
        DomainName.append('-')
        DomainEnd.append('-')
len(Domain)


# In[119]:

from collections import Counter
#domainName
counts = Counter(DomainName)
#print(counts)


# In[120]:

#localpart, emailLength
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


# In[131]:

# print(len(phoneNumb))
# print(len(cellNumb)) 
# print(len(firstName))
# print(len(lastName))
# print(len(Domain))
# print(len(DomainName))
# print(len(DomainEnd))
# print(len(localPart))
#print(len(emailLength))


# In[122]:

new_df = pd.DataFrame({'FName':firstName, 'LName':lastName,'Phone': phoneNumb, 'Cell': cellNumb, 
        'Email_Total_Length': emailLength, 'LocalPart': localPart,
                       'Domain': Domain, 'Domain_Name':DomainName,'Domain_End':DomainEnd})
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


# In[123]:

#emails['EMAIL_ADDR_DESC'][90341]


# In[124]:

#checknow
#new_df.LocalPart[24098:24100]


# In[125]:

#n_df.LocalPart[10:13]


# In[126]:

#len(new_df.LocalPart)


# In[127]:

# for i in new_df.LocalPart:
#     print(i)


# In[92]:

# print(len(LP_length))
# print(len(stopcount))
# print(len(numbercount))
# print(len(lettercount))
# print(len(fn_Match)) 
# print(len(ln_Match))
#90362


# In[93]:

new_df = pd.DataFrame({'App_Date': emails['app_date'], 'State': emails['PRIM_STATE_CD'], 'DOB': DOB_yr, 
                       'Address_MM':emails['addr_mm'],'FName':firstName, 'LName':lastName,'Phone': phoneNumb, 'Cell': cellNumb, 
        'Email_Total_Length': emailLength, 'LocalPart': localPart, 'LP_length': LP_length,
                       'Domain': Domain, 'Domain_Name':DomainName,'Domain_End':DomainEnd,'stops':stopcount, 
'FN_match':fn_Match, 'LN_match':ln_Match, 'LP_lettercnt':lettercount, 'LP_numbercnt':numbercount, 'xFraud_App': emails['fraud_app']})

#new_df.head(4)


# In[94]:

#emails.columns


# In[95]:

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
        #int('1')

new_df = pd.DataFrame({'Address': emails['PRIM_ADDR_DESC'],'App_Date': emails['app_date'], 'State': emails['PRIM_STATE_CD'], 'DOB': DOB_yr, 
                       'Address_MM':emails['addr_mm'],'FName':firstName, 'LName':lastName,'Phone': phoneNumb, 'Cell': cellNumb, 
        'Email_Total_Length': emailLength, 'LocalPart': localPart, 'LP_length': LP_length,
                       'Domain': Domain, 'Domain_Name':DomainName,'Domain_End':DomainEnd,'stops':stopcount,'FN_match':FirstName_match, 'LN_match':LastName_match, 'LP_lettercnt':lettercount, 'LP_numbercnt':numbercount, 'xFraud_App': emails['fraud_app']})


#new_df.head(4)


# In[96]:

new_df['numbers'] = new_df.LP_numbercnt/new_df.LP_length
#LP_numbercnt/LP_length
new_df['stopz'] = new_df.stops/new_df.LP_length
#stops/LP_length
new_df['letters'] = new_df.LP_lettercnt/new_df.LP_length
#LP_lettercnt/LP_length
new_df['FN_and_LN_match'] = new_df['FN_match'] + new_df['LN_match']
new_df['FN_and_LN_match'].head(4)


# In[97]:

new_df['FN_LN_match'] = np.where(new_df['FN_and_LN_match']==2, 1, 0)


# In[98]:

del new_df['FN_and_LN_match']


# In[99]:

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


# In[100]:

cell_area_code_state = cell_area_code.merge(areaCode_state, on='Area code', how='left')
area_code_state =area_code.merge(areaCode_state, on='Area code', how='left')
new_df['area_code_state'] = area_code_state['State code']
new_df['cell_area_code_state'] = cell_area_code_state['State code']


# In[130]:

new_df.shape


# In[102]:

new_df['area_code_state'] = new_df['area_code_state'].fillna('-')
new_df['cell_area_code_state'] = new_df['area_code_state'].fillna('-')
pd.set_option('display.max_columns', None)
new_df.head(4)


# In[103]:

newer_df=new_df[new_df['area_code_state'] != '-']
#rows where phone area code is present, retrieve the index, and merge 
#newer_df.head(4)
#len(newer_df) #53923 down from 90362


# In[104]:

#newer_df['cell_area_code_state'].equals(newer_df['area_code_state'])
newer_df['que'] = newer_df['State'] == newer_df['area_code_state'] #T/F
newer_df['cell_que'] = newer_df['State'] == newer_df['cell_area_code_state']
newer_df[newer_df['que'] == False].head(24)
newer_df[newer_df['que'] == False].shape #1423 total no matches with state and areacode
newer_df.shape #(53923, 28)


# In[105]:

newest_df = newer_df.que
#newest_df.index


# In[106]:

newest_df = pd.DataFrame({'que': newest_df})
#newest_df.head(10)


# In[107]:

emails_df = new_df.merge(newest_df, left_index=True, right_index=True, how='left')
emails_df['que'].fillna('-', inplace=True)
emails_df = emails_df.rename(index=str, columns={"que": "Phone_State_Match"})
#emails_df.head(4)


# In[108]:

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
        
len(Age)


# In[109]:

emails_df['Age_bucket'] = Age
emails_df['Age'] = 2016 - emails_df['DOB']


# In[110]:

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
emails_df.head(3)


# In[111]:

#Metrics
#emails_df[6500:6508]


# In[112]:

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
flse = newer_df[newer_df.que == False]
flse[flse['xFraud_App'] == 1] #(15, 28)


# In[113]:

emails_df.columns


# In[114]:

emails_df.columns


# In[115]:

emails_data = emails_df
emails_data = emails_data.drop(['Address_MM', 'App_Date', 'Cell', 'cell_area_code_state'], axis=1, inplace=True)


# In[116]:

type(emails_data)


# In[117]:

#emails_data = emails_df
#emails_data = emails_data.drop(['Address_MM', 'App_Date', 'Cell', 'cell_area_code_state'], axis=1, inplace=True)


# In[118]:

emails_df.head(3)


# In[119]:

emails_df.shape


# In[120]:

#emails_df.shape


# In[121]:

emals_df.shape


# In[122]:

emails_df.columns


# In[123]:

emals_df.columns


# In[124]:

del emals_df['FN_and_LN_match']


# In[125]:

#1/19
comp_df['addr_numbercount'] = addr_numbercount
comp_df['addr_wordLength'] = addr_wordLength
comp_df['addr_letter_count'] = addr_letter_count
indx = comp_df.index.tolist()
comp_df['index'] = indx
comp_df['match'].value_counts() #613
comp_df.head(3)
comp_df.shape #90362, 7

emails_df.shape #(90362, 25)
emails_df['index'] = emails_df.index.tolist()
emails_df['index']= emails_df['index'].apply(str)
comp_df['index']= comp_df['index'].apply(str)
result = pd.merge(emails_df, comp_df, on='index')
result.shape
result.head(2)

#1/19
#concerned about index b/c of break off 

#append two result tables together 
#result.shape #(8009, 31)
#90362, 32
#Phone we be dropped later


# result.columsn
# Index(['DOB', 'Domain', 'Domain_End', 'Domain_Name', 'Email_Total_Length',
#        'FN_match', 'FName', 'LN_match', 'LName', 'LP_length', 'LP_lettercnt',
#        'LP_numbercnt', 'LocalPart', 'Phone', 'State', 'stops', 'xFraud_App',
#        'numbers', 'stopz', 'letters', 'FN_LN_match', 'area_code_state',
#        'Phone_State_Match', 'Age_bucket', 'Age', 'index', 'addr_numbers',
#        'localpart_numbers', 'match', 'addr_numbercount', 'addr_wordLength',
#        'addr_letter_count'],
#       dtype='object')


# In[126]:

#result.shape #(90362, 32)


# In[127]:

emails_result_df = result
emails_result_df.shape


# In[128]:

##df = emails_df.append(emals_df)
#VSUM: emals dataframe #lds: emails dataframe
#both have result dataframes, renamed
#df: appended together

df = emails_result_df.append(emals_result_df)

df.shape

#1/19


# In[129]:

df.columns


# In[130]:

len(df.index.unique()) #90362

len(df.index) #98371


# In[131]:

df.head(3)


# In[132]:

#df.reset_index(drop=True)
#not sure if this did anything!
#comment out?


# In[133]:

#del df['FN_and_LN_match']
del df['Phone']


# In[134]:

test = df['Domain'].str.upper()

test[90350:90366]
test = test.reset_index()
test['Domain'].apply(lambda x: x.lower()).head(3)


# In[135]:

df['Domain'] = df['Domain'].apply(lambda x: x.upper())
df['Domain_End'] = df['Domain_End'].apply(lambda x: x.upper())
df['Domain_Name'] = df['Domain_Name'].apply(lambda x: x.upper())
df['FName'] = df['FName'].apply(lambda x: x.upper())
df['LName'] = df['LName'].apply(lambda x: x.upper())
df['LocalPart'] = df['LocalPart'].apply(lambda x: x.upper())
df[90350:90366]

#emals has NaN for Phone, 
#caps for FNAME, LNAME, LocalPart, Domain, Domain_End, Domain_Name, lower case -> upper?
#remove FN_and_LN_match column
#drop phone


# In[136]:

df.shape #(98371, 31)
df.columns


# In[137]:

#1/19
df.groupby('xFraud_App').match.value_counts()


# In[138]:

emails_result_df.groupby('xFraud_App').match.value_counts() 
#may need more samples? in the cvs pull there's no matches for fraud_app=1


# In[139]:

#89212+613+537


# In[140]:

emals_result_df.groupby('xFraud_App').match.value_counts()


# In[141]:

df.groupby('xFraud_App').Age_bucket.value_counts()


# In[142]:

#emails_df.groupby('xFraud_App').Address_MM.value_counts()


# In[143]:

emails_df.groupby('xFraud_App').Age_bucket.value_counts()


# In[144]:

df[df['Domain'] == '_._']  #1/23


# In[145]:

df.groupby('xFraud_App').State.value_counts()


# In[146]:

df.groupby('xFraud_App').FN_LN_match.describe()


# In[147]:

df.groupby('xFraud_App').stops.describe()


# In[148]:

df.groupby('xFraud_App').numbers.describe() 


# In[149]:

df.xFraud_App.value_counts()

#8546/89825


# In[150]:

df[df.xFraud_App == 0].Domain.value_counts().nlargest(15)

#-.-


# In[151]:

df[df.xFraud_App == 1].Domain.value_counts().nlargest(15) #same one over time 
#_._  


# In[152]:

df.head(3)


# In[153]:

df.shape


# In[154]:

#df.ix[df.Domain==['_._'|'-.-'], 'Domain_End'] = 'NaN'



# df[df['Domain_Name'] == 'NaN']


# In[156]:

#df[df['Domain_Name'] == '-'].shape #8310... to 21
#df.shape #98371
#df[df['Domain_Name'] == '-']  ##MARTIN.MCGUNNBNYMELLON.COM...104240
#missing @ symbol, refused, none


#-.-
#1/23


# In[157]:

#domain buckets based on frequency
freqtable = df[df.xFraud_App == 1].groupby(["Domain_Name"]).size().reset_index(name="cnt")
freqtable_non = df[df.xFraud_App == 0].groupby(["Domain_Name"]).size().reset_index(name="cnt")

#frequency as a percentage of total count
#1) count frequency
#2) count total
#3) perc, new column
#4) find mean, min values
#5) bucketing... 

#1/23


# In[162]:

df.shape

#export df to another notebook due to size of this on being so slow


# In[217]:

df.Domain_Name.unique().shape  #5231... looking for unique values, not all. 


# In[200]:

df.head(2)


# In[ ]:

#Feb 2017
#Domain, Target (1, 0), niY: Total Count of Target = 1 per Domain, P(Y): niY/totalRows, 
#P(niY=1)/ni: niY/totalRows of domain, 
#lambda = 1/(1+ e^(-(niY - k)/f))
#k= 20, f = 4
#S = lambda*P(Y) + (1-lambda)*P(niY=1)/ni
#EXP(-(4-20)/4)

#niY=1...group by domain, and sum number of rows where Y=1 
#freqtable: Domain_Name, cnt .....freqtable = df[df.xFraud_App == 1].groupby(["Domain_Name"]).size().reset_index(name="cnt")

#totalRows = nrows(df): 98371

#totalRowsDomian ...group by domain, sum total number of rows
#df.groupby(["Domain_Name"]).size().reset_index(name="total_cnt").head(30)
#5231 unique rows or domains

#lambda, S

#mapping table to dataset or substitute per domain


# In[215]:

freqtable.shape  #518
#freqtable.cnt
#fraud only

#freqtable_non.shape  #4840


# In[228]:

lookup = df.groupby(["Domain_Name"]).size().reset_index(name="total_cnt")
a_c =lookup.merge(freqtable, on='Domain_Name', how='left')

a_c.shape #5231

a_c


# In[218]:

freqtable.head(20)


# In[210]:

lookup = df.groupby(["Domain_Name"]).size().reset_index(name="total_cnt")
lookup.total_cnt.shape #5231
lookup.head(20)


# In[204]:

df.groupby(["Domain_Name"]).size().reset_index(name="total_cnt").shape


# In[ ]:

freqtable['cnt']/lookup['total_cnt']

#What about domains with no fraud
#What about new domains in testing set
#recalculate probabiliyt of Y=1 in new dataset/
#Write a few functions. 

#CreateMapping
# if null, create based off of the new dataset
#otherwise, pull it for the old dataset


# In[ ]:

#5231-518 =  ;  unique domains that don't have fraud for this set. merge.. may be 0. screw up S?
#need to make a new table.. per domain..P(Y), and count total occruences.  
#merge on domain... 


# In[ ]:

#lambda
#k = 20
#f = 4
#niY = look up freqtable for domain ni Y total cnt 
#lambda = 1/(1+e^(-(niY-k)/f))

#S perDomain = (lambda * freqtable['Domain_Name' == i]) + (l-lambda)* freqtable['cnt']/lookup['total_cnt']


# In[164]:

#freqtable_non.sort(['cnt'])
result = freqtable.merge(freqtable_non, on='Domain_Name', how='left') #joined to compare
result.head(3)
#x 
#fraud - gmail: 2416 for what period of time..28%/ hotmail:358..0.04%/yahoo:1644..0.19%/Comcast: 84...0.098%

#periods, dashes in domain
#numbers in domain

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
        
#df['numbercount_domain'] = numbercount_domain
#df['stopcount_domain'] = stopcount_domain
#df['lettercount_domain'] = lettercount_domain

#df['numbers_domain'] = df.numbercount_domain/df.domain_length
#df['stopz_domain'] = df.stopcount_domain/df.domain_length
#df['letters_domain'] = df.lettercount_domain/df.domain_length

#vowels

vowel_count = []
vowels = set("AEIOU")
for domain in df.Domain_Name: #letter vs. word...GMAIL
    count = 0
    vowlz=[]
    for letter in domain:#..G...M...A...I...L
        if letter in vowels:
            #vowlz=[]
            vowlz.append(letter)
    vowel_count.append(len(vowlz))
    #print(vowlz, len(vowlz))
            #count += 1
            #for number in range(0, count):
                #print(domain,count)            
            #print(domain,count) #per word, no sum, no max
            #vowel_count.append(count)
        
#"Mail"

#still have frequency as category

#y 
#nonfraud - gmail: 31472... 35%/hotmail:7104..0.08%/yahoo:17808..0.19%/Comcast: 2303..0.025%
#variations of gmail..same misspellings occur in both

#vowel_count[0:15] #[1, 2, 1, 2, 1, 2, 3, 1, 2, 1, 2, 1, 2, 3, 1]
len(vowel_count)


# In[165]:

df['domain_vowel_count'] = vowel_count


# In[166]:

#df[90370:90380]


# In[167]:

#df[df['Domain'] == '_._']


# In[168]:

df.shape


# In[169]:

emails_result_df.shape


# In[170]:

emals_result_df.shape


# In[171]:

#export emals
#emals_result_df.to_csv("emals.csv")
from pandas import ExcelWriter
writer = ExcelWriter('emals.xlsx')
emals_result_df.to_excel(writer,'Sheet1')
writer.save()

# In[172]:

#export emails
#emails_result_df.to_csv("emails.csv")

from pandas import ExcelWriter
writer = ExcelWriter('emails.xlsx')
emails_result_df.to_excel(writer,'Sheet1')
writer.save()



# In[229]:

#export df
#df.to_csv("df.csv")
from pandas import ExcelWriter
writer = ExcelWriter('df.xlsx')
df.to_excel(writer,'Sheet1')
writer.save()

len(lettercount_domain)
# In[230]:

#freqtable.to_csv("freqtable.csv")
from pandas import ExcelWriter
writer = ExcelWriter('freqtable.xlsx')
freqtable.to_excel(writer,'Sheet1')
writer.save()

#freqtable_non.to_csv("freqtable_non.csv")
from pandas import ExcelWriter
writer = ExcelWriter('freqtable_non.xlsx')
freqtable_non.to_excel(writer,'Sheet1')
writer.save()

#lookup.to_csv("lookup.csv")
from pandas import ExcelWriter
writer = ExcelWriter('lookup.xlsx')
lookup.to_excel(writer,'Sheet1')
writer.save()

# In[174]:

freqtable['perc'] = freqtable.cnt/sum(freqtable.cnt)
freqtable['perc'].describe()

fraud_freq_mean = freqtable['perc'].mean()
fraud_freq_min = freqtable['perc'].min()

# for i in freqtable['perc']:
#     if i  > freqtable['perc'].mean():
#         print(i)

domainBucket=[]
for i in freqtable['perc']:
    #print(i)
#     if i == '_':
#         print(i)
#         domainBucket.append('Null')
    if i == freqtable['perc'].mean():
        domainBucket.append('Avg')  #should be mean or median of cnt frequency of occurance
    elif i > freqtable['perc'].mean() and i < (freqtable['perc'].mean() + freqtable['perc'].std()):
        domainBucket.append('Frequent')
    elif i > (freqtable['perc'].mean() + freqtable['perc'].std()) and i < (freqtable['perc'].mean() + 2*freqtable['perc'].std()):
        domainBucket.append('More Frequent')
#     elif i >=30 and i < 50:
#         domainBucket.append('High')
#     elif i >=50:
#         domainBucket.append('Very High')
    else:
        domainBucket.append('Below_Avg')

domainBucket[0:3]
freqtable.head(3)


# In[175]:

freqtable_non['perc'] = freqtable_non.cnt/sum(freqtable_non.cnt)
freqtable_non['perc'].describe()


# In[176]:

#freqtable_non.sort(['cnt'])


# In[177]:

freqtable = df[df.xFraud_App == 1].groupby(["Domain_Name"]).size().reset_index(name="cnt")
freqtable_non = df[df.xFraud_App == 0].groupby(["Domain_Name"]).size().reset_index(name="cnt")
#maybe should be other way around - compare with non-fraud

#delete row with '-' from dataframe
#join with df based on ---Domain/Domain_End/Domain_Name/LocalPart  - yes
#1/19
domainBucket=[]
for i in freqtable['cnt']:
    #print(i)
#     if i == '_':
#         print(i)
#         domainBucket.append('Null')
    if i == 1:
        domainBucket.append('low')  #should be mean or median of cnt frequency of occurance
    elif i > 1 and i < 5:
        domainBucket.append('lowMed')
    elif i >=5 and i < 30:
        domainBucket.append('Med')
    elif i >=30 and i < 50:
        domainBucket.append('High')
    elif i >=50:
        domainBucket.append('Very High')
    else:
        domainBucket.append('new?')
                
#bucketing by risk based on eda metrics? fraudapp = 1 here. 
#freqtable
#apply to original dataset being fed in, and to final dataset


# In[178]:

#domainBucket
len(domainBucket) #507
len(freqtable.Domain_Name) #518
freqtable['domainBucket'] = domainBucket
freqtable.head(5)


# In[179]:

r = pd.merge(df, freqtable, on='Domain_Name')


# In[180]:

r[r['domainBucket'] == 'low'].xFraud_App.value_counts() #even tho its low, 305 is still fraud


# In[181]:

r[r['domainBucket'] == 'low']


# In[182]:

cnts = df[df.xFraud_App == 1].Domain.value_counts().nlargest(15)
dict(cnts)


# In[183]:

#emails_df.groupby('xFraud_App').Domain.value_counts()
emails_df[emails_df.xFraud_App == 1].Domain.value_counts()


# In[184]:

#df.groupby('xFraud_App').numbers.describe() 

df.groupby('xFraud_App').Phone_State_Match.value_counts()


# In[185]:

5055/89825


# In[186]:

1321/8546


# In[187]:

emals_df.head(5)


# In[188]:

del df['FN_and_LN_match']
del df['Address']
del df['index']
df.head(3)


# In[189]:

#Common domains
#Common States


# In[1]:

df.shape #31 or 33 columns?


# In[191]:

#Final Features - 1/19

#export final dataset -> python35 tectia for random forest 

#Response Variable
    #xFraud_App = 0, 1
    
#Categorical to dummify
    #Age_bucket
    #State bucketing?
    
    #Domain bucketing? = top 5, 10, 15, 20, etc. ..common vs. uncommon

#correlated
    #Email_Total_Length
    #LP_length
    #LP_lettercnt
    #LP_numbercnt
    #stops (count of total stops)
    #letters (proportion of total email is comprised of this)
    #numbers
    #stopz
    
#Address
    #addr_letter_count
    #addr_numbercount
    #addr_numbers
    #addr_wordLength
    
#match
    #exact number match between address and email
    
#first/last name match local part
    #FN_LN_match
    #FN_match
    #LN_match
    
#Phone and State
    #Phone_State_Match
    

#initials match with email
    #first letter from fname, first letter from lname
    #add together as W.W. or WW 
    #middle initial, etc. 
    #localpart W.W. with periods or WW with no periods
    #search localpart for that
    #match = Y/N
    
#various john patterns - some way to automate combos
#FN|MI|LAST NAME INIT|2 digits from house number 


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




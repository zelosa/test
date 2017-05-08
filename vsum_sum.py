import matplotlib
matplotlib.use('Agg')
import re
import numpy as np
import pandas as pd
#import matplotlib
#get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
#from __future__ import division
from scipy import stats


# In[2]:

#read in #VSUM Daily Reports
emals= pd.read_excel('emals.xlsx') #, header = 1)
emails= pd.read_excel('emails.xlsx')
df= pd.read_excel('df.xlsx')
emals_= pd.read_excel('emals.xlsx') #, header = 1)
emails_= pd.read_excel('emails.xlsx')


# In[3]:

df_= pd.read_excel('df.xlsx')


# In[4]:

#much more non-fraud than fraud.
lookup_=pd.read_excel('lookup.xlsx') #ALL
freqtable_=pd.read_excel('freqtable.xlsx') #fraud
freqtable_non=pd.read_excel('freqtable_non.xlsx') #non-fraud

#freqtable = df[df.xFraud_App == 1].groupby(["Domain_Name"]).size().reset_index(name="cnt")
#freqtable_non = df[df.xFraud_App == 0].groupby(["Domain_Name"]).size().reset_index(name="cnt")
#lookup = df.groupby(["Domain_Name"]).size().reset_index(name="total_cnt")


# In[5]:

freqtable = freqtable_[['Domain_Name','cnt']] 


# In[6]:

data = df_[['Domain_Name', 'xFraud_App']]


# In[7]:

data.reset_index(inplace = True)


# In[8]:

del data['index']


# In[9]:

tst = data.merge(freqtable, on='Domain_Name', how='left')


# In[10]:

tst[tst['Domain_Name'] == 'ALEXOXO.33MAIL'].index == data[data['Domain_Name'] == 'ALEXOXO.33MAIL'].index


# In[11]:

g = tst['cnt']
data['cnt'] = g


# In[12]:

data['freqtableCNT_over_ALL'] = data['cnt']/len(data)


# In[13]:

c = data.merge(lookup_, on='Domain_Name', how='left')
c['freqtableCNT_over_lookup'] = c['cnt']/c['total_cnt']


# In[14]:

from math import exp
exp(1e-5) - 1

elambda = []
for i in c['freqtableCNT_over_ALL']:
    elambda.append(1/(1+exp(-(i-20)/4)))
    
len(elambda)
c['lambda']= elambda
#S = (lambda*freqtableCNT_over_ALL) + (1 - freqtableCNT_over_all)* freqtableCNT_over_lookup
c['S'] = (c['lambda']*c['freqtableCNT_over_ALL']) + ((1 - c['freqtableCNT_over_ALL'])*c['freqtableCNT_over_lookup'])


# In[15]:

cc = c[c.cnt.isnull()]
#cc[cc['xFraud_App'] == 0] #6389 rows 


# In[16]:

cc.shape


# In[17]:

#workflow


# In[18]:

#feed in c 
    #if c.cnt is null
        #add these new columns tracking Missingness.
    #else if not null
        #new columns will null entry.

missing = c[c.cnt.isnull()]
missing.total_cnt.sum() #32009

prob_miss = missing.total_cnt/missing.total_cnt.sum()
#same for all? 
prob_uniq_miss = 1/len(missing)
print(prob_uniq_miss)

missing['prob_miss'] = prob_miss
missing['prob_uniq_miss'] = 1/len(missing)

from math import exp
# exp(1e-5) - 1

elambdah = []
for i in missing['total_cnt']:
    elambdah.append(1/(1+exp(-(i-20)/4)))
    
missing['lambda']= elambdah
missing['Smiss'] = (missing['lambda']*missing['prob_miss']) + ((1 - missing['prob_miss'])*missing['prob_uniq_miss'])


# In[19]:

test_two = c.merge(missing, left_index=True, right_index=True, how='left')
test_two[16:18]


# In[20]:

newDf = test_two.loc[:,['Domain_Name_x','S_x','Smiss']]


# In[ ]:

newDff = test_two.loc[:,['Domain', 'Domain_Name_x','S_x','Smiss']]


# In[21]:

newDf[newDf.S_x.notnull()].sort_values(by='S_x', ascending=False).head(4)


# In[22]:

newDf.rename(columns={'Domain_Name_x': 'Domain_Name'}, inplace=True)


# In[23]:

df_.reset_index(inplace = True)


# In[24]:

test_two = df_.merge(newDf, left_index=True, right_index=True, how='left')


# In[25]:

del test_two['level_0']
test_two


# In[26]:

test_two.FN_LN_match.value_counts()


# In[27]:

#.loc should include Domain/Domain_Name
newDf = test_two.loc[:,['Age_bucket','Domain', 'Domain_End','Email_Total_Length', 'FN_LN_match', 'LP_length', 'Phone_State_Match', 'match', 'numbers', 'stopz', 'domain_vowel_count',
                        'S_x', 'Smiss', 'xFraud_App', 'letters', 'State', 'addr_numbercount', 'addr_wordLength', 'addr_letter_count']]


# In[28]:

# newDf = test_two.loc[:,['Age_bucket','Domain_End','Email_Total_Length', 'FN_LN_match', 'LP_length', 'Phone_State_Match', 'match', 'numbers', 'stopz', 'domain_vowel_count',
#                         'S_x', 'Smiss', 'xFraud_App']]


# In[29]:

#newDf[newDf['FN_LN_match'].isnull()].head(3)


# In[30]:

#newDf.FN_and_LN_match.value_counts()

#Are Unknowns/NA's the result of an incorrect join? 


# In[31]:

#sum(newDf.FN_and_LN_match.isnull())


# In[32]:

#sum(newDf.FN_and_LN_match.notnull())


# In[33]:

4297+2169+1543


# In[34]:

newDf.dtypes

for cat in ['Age_bucket','Domain_End','Email_Total_Length', 'FN_LN_match', 'LP_length', 'Phone_State_Match', 'match', 'numbers', 'stopz', 'domain_vowel_count',
                        'S_x', 'Smiss', 'xFraud_App']:
    print("Number of levels in category '{0}': \b {1:2.2f} ".format          (cat, newDf[cat].unique().size))


# In[35]:

freqtable_DE = df[df.xFraud_App == 1].groupby(["Domain_End"]).size().reset_index(name="cnt")
freqtable_non_DE = df[df.xFraud_App == 0].groupby(["Domain_End"]).size().reset_index(name="cnt")
lookup_DE = df.groupby(["Domain_End"]).size().reset_index(name="total_cnt")
data_DE = df_[['Domain_End', 'xFraud_App']]
data_DE.reset_index(inplace=True)
del data_DE['index']
tst = data_DE.merge(freqtable_DE, on='Domain_End', how='left')
g = tst['cnt']
data_DE['cnt'] = g
data_DE['freqtableCNT_over_ALL'] = data_DE['cnt']/len(data_DE)
c_DE= data_DE.merge(lookup_DE, on= 'Domain_End', how='left')
c_DE['freqtableCNT_over_lookup'] = c_DE['cnt']/c_DE['total_cnt']
elambda_DE = []
for i in c_DE['freqtableCNT_over_ALL']:
    elambda_DE.append(1/(1+exp(-(i-20)/4)))
c_DE['lambda']= elambda_DE
#S = (lambda*freqtableCNT_over_ALL) + (1 - freqtableCNT_over_all)* freqtableCNT_over_lookup
c_DE['S'] = (c_DE['lambda']*c_DE['freqtableCNT_over_ALL']) + ((1 - c_DE['freqtableCNT_over_ALL'])*c_DE['freqtableCNT_over_lookup'])
c = c_DE
missing = c[c.cnt.isnull()]
missing.total_cnt.sum() #32009
prob_miss = missing.total_cnt/missing.total_cnt.sum()
prob_uniq_miss = 1/len(missing)
print(prob_uniq_miss)
missing['prob_miss'] = prob_miss
missing['prob_uniq_miss'] = 1/len(missing)

elambdah = []
for i in missing['total_cnt']:
    elambdah.append(1/(1+exp(-(i-20)/4)))
missing['lambda']= elambdah
missing['Smiss'] = (missing['lambda']*missing['prob_miss']) + ((1 - missing['prob_miss'])*missing['prob_uniq_miss'])
test_two = c.merge(missing, left_index=True, right_index=True, how='left')
newDf_DE = test_two.loc[:,['Domain_End_x','S_x','Smiss']]
#df_.reset_index(inplace = True)df_.reset_index(inplace = True)df_.reset_index(inplace = True)
test_two = df_.merge(newDf_DE, left_index=True, right_index=True, how='left')
del test_two['level_0']
newDf_DE = test_two.loc[:,['Age_bucket','Domain_Name', 'Domain_End','Email_Total_Length', 'FN_LN_match', 'LP_length', 'Phone_State_Match', 'match', 'numbers', 'stopz', 'domain_vowel_count',
                        'S_x', 'Smiss', 'xFraud_App']]
newDf_DE.rename(columns={'Domain_End_x': 'Domain_End','S_x': 'DomainEnd_Sx', 'Smiss': 'Smiss_DE' }, inplace=True)
newDf_DE = newDf_DE.loc[:,['Domain_End','DomainEnd_Sx','Smiss_DE']]


# In[36]:

#START HERE
test_df = newDf.merge(newDf_DE, left_index=True, right_index=True, how='left')
test_df.head(1)


# In[37]:

for cat in ['Age_bucket','Domain_End_x','Email_Total_Length', 'FN_LN_match', 
    'LP_length', 'Phone_State_Match', 'match', 'numbers', 'stopz', 'domain_vowel_count','S_x', 'Smiss', 'xFraud_App', 'DomainEnd_Sx','Smiss_DE']:
    print("Number of levels in category '{0}': \b {1:2.2f} ".format          (cat, test_df[cat].unique().size))


# In[38]:

Age_bcket = pd.get_dummies(test_df.Age_bucket.fillna('UnkAge'))
name_match = pd.get_dummies(test_df.FN_LN_match.fillna('UnkNameMatch'))
P_S_match = pd.get_dummies(test_df.Phone_State_Match.fillna('UnkPSMatch')) #replace '-'? 
#match_Adr_LP = pd.get_dummies(test_df.match)


# In[39]:

data_new = test_df
data_new.drop(['Age_bucket', 'Domain_End_x', 'Domain_End_y', 'Phone_State_Match','match'], axis=1, inplace=True)


# In[40]:

data_new.head(3)


# In[41]:

for column in data_new:
    if column in ('S_x', 'Smiss','DomainEnd_Sx', 'Smiss_DE'):
        mean_value = int(float(data_new[column].mean(skipna=True, axis=0)))
        data_new[column].fillna(mean_value, inplace=True)


# In[42]:

data_new.head(3)


# In[43]:

data_new = pd.concat([data_new, Age_bcket, name_match, P_S_match], axis=1)


# In[44]:

name_match.head(3)


# In[45]:

data_new.columns


# In[46]:

data_new.head(2)


# In[47]:

int(float(data_new['Smiss_DE'].mean(skipna=True, axis=0)))


# In[48]:

print(data_new.isnull().sum()) 


# In[49]:

data_new.shape


# In[50]:

#save as xksx
data_new.to_csv("data_new_march.csv")

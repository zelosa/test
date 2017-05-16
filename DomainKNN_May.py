


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


# In[3]:

#3/22
Domain_Sx= pd.read_csv('data_new_march.csv') #data_new_march from vsum_sum.py


# In[4]:

#3/22
Domain_Sx.columns


# In[ ]:

#Domain_Sx= pd.read_excel('Feb_data_new.xlsx') #data_new_march from vsum_sum.py


# In[3]:

#Domain_Sx.columns


# In[5]:

Domain_Sx = Domain_Sx.loc[:,['Domain','S_x','DomainEnd_Sx', 'xFraud_App']]


# In[6]:

#len(Domain_Sx.Domain.unique()) #5414


# In[7]:

df = Domain_Sx.loc[:,['Domain','S_x','DomainEnd_Sx']]
df.duplicated()
df = df.drop_duplicates()


# df.head(3) #df_Sx.xlsx

# In[ ]:

#3/22
#what feeds into what
#SAS source

werking_output= pd.read_excel('emails_may.xlsx') #from werking.py file, emails.xlsx
#emails_march.xlsx

# In[9]:

print('werking_output for may shape', werking_output.shape)


# In[11]:

#werking_output_224.columns


# In[13]:

werking_output.head(1) #NaN


# In[14]:

df.head(3)
print(df.shape)

# In[15]:

from pandas import ExcelWriter
writer = ExcelWriter('df_Sx_May.xlsx') #there's also alot of Sx = 0. DomainEnd_Sx equal to Sx? 
#df_Sx.xlsx
df.to_excel(writer,'Sheet1')
writer.save()


# In[ ]:

#3.22
a_c_state =df.merge(werking_output, on='Domain', how='left')


# In[17]:

a_c_state.shape #(35219, 33)


# In[18]:

werking_output.shape #what does it do w/ all the singles? that don't have Sx? different join? 


# In[19]:

df.shape #no duplicates


# In[21]:

c_state =df.merge(werking_output, on='Domain', how='right') #right vs. left?


# In[39]:

#c_state.S_x.value_counts()


# In[42]:

(c_state.S_x.isnull().sum()/c_state.S_x.notnull().sum())*100 #4.4% missing l


# In[22]:

# c_state.tail(20)
# S_x
c_st = c_state[pd.isnull(c_state['S_x'])] #1333 rows 


# In[43]:

c_st.columns


# In[91]:

NaN_Sx = c_state.loc[:,['Domain','S_x','Domain_Name', 'Domain_End']]


# In[47]:

from pandas import ExcelWriter
writer = ExcelWriter('NaN_Sx_May.xlsx') #there's also alot of Sx = 0. DomainEnd_Sx equal to Sx? 
#NaN_Sx.xlsx
NaN_Sx.to_excel(writer,'Sheet1')
writer.save()


# In[48]:

NaN_Sx.columns


# In[65]:

NaN_Sx['Domain_Name'].fillna('-', inplace=True)

NaN_vowel_count = []
NaN_cons_count = []
vowels = set("AEIOU")
consonants = list("BCDFGHJKLMNPQRSTVEXZ")
for domain in NaN_Sx.Domain_Name: #letter vs. word...GMAIL
    count = 0
    vowlz=[]
    conz =[]
    if type(domain) != str:
        print(domain, type(domain))
    else:
        for letter in domain:#..G...M...A...I...L
            if letter in vowels:
            #vowlz=[]
                vowlz.append(letter)
            elif letter in consonants:
                conz.append(letter)
        NaN_vowel_count.append(len(vowlz))
        NaN_cons_count.append(len(conz))    


# In[59]:

#NaN_vowel_count[45:50]


# In[66]:

#NaN_cons_count[45:50]


# In[67]:

NaN_domain_length = []
for i in NaN_Sx['Domain_Name'].dropna():
    #print(len(i))  #str, float 
    NaN_domain_length.append(len(i))  


# In[68]:

#NaN_domain_length[1:10]


# In[72]:

#for i in [NaN_Sx, NaN_vowel_count, NaN_cons_count, NaN_domain_length]:
    #print(type(i))


# In[78]:

#len(NaN_vowel_count) #31448


# In[86]:

NaN_Sx.shape

df1 = pd.DataFrame({'vowels': NaN_vowel_count, 'constants': NaN_cons_count,'Domain_Length': NaN_domain_length})
df1.head(2)

#dummify domain end


# In[96]:

DE_NaN = pd.get_dummies(NaN_Sx.Domain_End)


# In[97]:

#NaN_Sx.columns
#d = pd.concat([NaN_Sx, NaN_vowel_count, NaN_cons_count, NaN_domain_length], axis=1)
d = pd.concat([NaN_Sx, DE_NaN, df1], axis=1)
d.head(2)


# In[129]:

d.columns


# In[130]:

df = d


# In[142]:

df.head(3)
#print(df.shape)

# In[141]:

del df['Domain_End']


# In[168]:

df.columns # used for KNN


# In[164]:

#dfe.shape


# #KNN

# In[30]:

#3.22
lizt = df.columns.tolist() 
#XX = df[[lizt]].as_matrix() unhashable
XX = df[lizt]
X = XX.as_matrix()
print(X.shape)


# In[173]:
#Error to fix, nothing below works? try IPython version?
np.isnan(X) #3.22 XX


# In[186]:


# In[ ]:

#3.22
tst = X


# In[118]:

import numpy as np
import random
from sklearn import datasets
from sklearn import neighbors

def impute(mat, learner, n_iter=3):
    mat = np.array(mat)
    mat_isnan = np.isnan(mat)        
    w = np.where(np.isnan(mat))
    ximp = mat.copy()
    for i in range(0, len(w[0])):
        n = w[0][i] # row where the nan is
        p = w[1][i] # column where the nan is
        col_isnan = mat_isnan[n, :] # empty columns in row n
        train = np.delete(mat, n, axis = 0) # remove row n to obtain a training set
        train_nonan = train[~np.apply_along_axis(np.any, 1, np.isnan(train)), :] # remove rows where there is a nan in the training set
        target = train_nonan[:, p] # vector to be predicted
        feature = train_nonan[:, ~col_isnan] # matrix of predictors
        learn = learner.fit(feature, target) # learner
        ximp[n, p] = learn.predict(mat[n, ~col_isnan].reshape(1, -1)) # predict and replace
    for iter in range(0, n_iter):
        for i in random.sample(range(0, len(w[0])), len(w[0])):
            n = w[0][i] # row where the nan is
            p = w[1][i] # column where the nan is
            train = np.delete(ximp, n, axis = 0) # remove row n to obtain a training set
            target = train[:, p] # vector to be predicted
            feature = np.delete(train, p, axis=1) # matrix of predictors
            learn = learner.fit(feature, target) # learner
            ximp[n, p] = learn.predict(np.delete(ximp[n,:], p).reshape(1, -1)) # predict and replace
    
    return ximp

# Impute with learner in the iris data set
iris = datasets.load_iris()
mat = iris.data.copy()

# throw some nans
mat[0,2] = np.NaN
mat[0,3] = np.NaN
mat[1,3] = np.NaN
mat[11,1] = np.NaN
mat = mat[range(30), :]

# impute
impute(mat=mat, learner=neighbors.KNeighborsRegressor(n_neighbors=3), n_iter=10)


# In[188]:

K = impute(mat=tst, learner=neighbors.KNeighborsRegressor(n_neighbors=3), n_iter=10)


# In[196]:

K.shape


# In[185]:

#impute(mat=tst, learner=neighbors.KNeighborsRegressor(n_neighbors=3), n_iter=10)


# In[197]:

print(X[31446:31448])


# In[199]:

print(K[1446:1448]) #worked!


# In[98]:

from pandas import ExcelWriter
writer = ExcelWriter('d_May.xlsx') #there's also alot of Sx = 0. DomainEnd_Sx equal to Sx? 
#d_2.xlsx
d.to_excel(writer,'Sheet1')
writer.save()


# In[117]:

# import pip #needed to use the pip functions
# for i in pip.get_installed_distributions(local_only=True):
#     print(i)


# In[100]:

#KNN for missing S_x
#Or could just make all the new scores 0. 

d.S_x.value_counts()


# In[102]:

d.S_x.isnull().sum()


# In[24]:

c_st.head(3) #fresh data with no Domain match with training data to calculate S_x score. 
#calculate score based on other features?


# In[25]:

c_st.Phone_State_Match.value_counts()


# In[26]:

#c_st[c_st['Phone_State_Match'] == False] #LHWOLVES.NET 


# In[27]:

#c_st.Domain.value_counts() #Either could be new client trend or new fraud trend


# In[28]:

#werking_output.Domain.value_counts()


# In[29]:

#31448-30115 #werking_outpute.shape - a_c_state_nn.shape where address is NaN from left join


# In[30]:

#35219-30115 #werking_output has 5K new domains previously unseen. how to isolate? 


# In[31]:

a_c_state.tail(200) #NaNs when fresh data doesn't have it present
#remove rows with NaNs for  Address


# In[32]:

a_c_state_nn = a_c_state[pd.notnull(a_c_state['Address'])]


# In[33]:

a_c_state_nn.shape


# In[35]:

a_c_state_nn.shape #(30115, 33)

from pandas import ExcelWriter
writer = ExcelWriter('a_Sx_May.xlsx') #there's also alot of Sx = 0. DomainEnd_Sx equal to Sx? 
#a_Sx_2.xlsx
a_c_state_nn.to_excel(writer,'Sheet1')
writer.save()


# In[ ]:

a_c_state


# In[ ]:

df.shape
df.dtypes


# In[ ]:

werking_output['Domain'].dtypes


# In[ ]:

# import numpy as np
# def parse(x):
#     try:
#         return str(x)
#     except ValueError:
#         return np.nan

# df['Domain'] = df['Domain'].apply(parse)


# In[ ]:

# werking_output['Domain'] = werking_output['Domain'].apply(parse)


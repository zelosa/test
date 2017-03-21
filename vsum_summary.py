
# coding: utf-8

# In[1]:

#read in libraries
#get_ipython().magic('pwd')
#get_ipython().magic('cd C:\\Users\\ww32293\\Desktop')
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

newDf[newDf['FN_LN_match'].isnull()].head(3)


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
data_new.drop(['Age_bucket', 'Domain_End_x', 'Domain_End_y', 'FN_LN_match', 'Phone_State_Match','match'], axis=1, inplace=True)


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
data_new.to_csv("data_new.csv")


# In[51]:

data_new.columns


# In[ ]:




# In[52]:

#read in data_new
data_new= pd.read_excel('data_new.xlsx') 


# In[53]:

target = test_df['xFraud_App']


# In[54]:

data_new.drop(['xFraud_App'], axis=1, inplace=True)


# In[55]:

data_new.shape


# In[56]:

#print(data_new.isnull().sum())
data_new.columns


# In[57]:

from sklearn.cross_validation import train_test_split
has_driven = pd.Series(target)
X_train, X_test, Y_train, Y_test = train_test_split(data_new,                                        has_driven,test_size=0.33,random_state=0) #65908, 32463
print('Train data set ratio:', sum(Y_train)/float(X_train['Email_Total_Length'].count()))
print('Test data set ratio:', sum(Y_test)/float(X_test['Email_Total_Length'].count()))


# In[58]:

from sklearn import ensemble
randomForest = ensemble.RandomForestClassifier(n_jobs = -1, oob_score=True) #adding oob
import sklearn.grid_search as gs 


# In[59]:

grid_para_forest = [{"n_estimators": [90], "criterion": ["gini"], 
                     'max_depth': [11], "min_samples_leaf": [3]}]
grid_search_forest = gs.GridSearchCV(randomForest, grid_para_forest,cv=5, scoring='accuracy').fit(X_train, Y_train)

print('The best score is %.4f' %grid_search_forest.best_score_)
print('The best parameters are %s'%grid_search_forest.best_params_)
print('The training error is %.4f'%(1 - grid_search_forest.best_estimator_.score(X_train, Y_train)))
print('The testing error is %.4f'%(1 - grid_search_forest.best_estimator_.score(X_test, Y_test)))


# In[60]:

grid_para_forest = [{"n_estimators": [10], "criterion": ["gini"], 
                     'max_depth': [5], "min_samples_leaf": [3]}]
grid_search_forest = gs.GridSearchCV(randomForest, grid_para_forest,cv=5, scoring='accuracy').fit(X_train, Y_train)

print('The best score is %.4f' %grid_search_forest.best_score_)
print('The best parameters are %s'%grid_search_forest.best_params_)
print('The training error is %.4f'%(1 - grid_search_forest.best_estimator_.score(X_train, Y_train)))
print('The testing error is %.4f'%(1 - grid_search_forest.best_estimator_.score(X_test, Y_test)))


# In[61]:

len(data_new.columns)


# In[62]:

list(range(10,110,10))


# In[63]:

grid_para_forest = [{"n_estimators": list(range(10, 110, 10)), "criterion": ["gini", "entropy"], 
                      'max_depth': list(range(10, 20)), "min_samples_leaf": list(range(3, 5))}]

grid_search_forest = gs.GridSearchCV(randomForest, grid_para_forest,cv=5, scoring='accuracy').fit(X_train, Y_train)

# #ValueError: Parameter values should be a list.


# #5 fold cross validation used
print('The best score is %.4f' %grid_search_forest.best_score_)
# print('The best parameters are %s'%grid_search_forest.best_params_)
# print('The training error is %.4f'%(1 - grid_search_forest.best_estimator_.score(X_train, Y_train)))
# print('The testing error is %.4f'%(1 - grid_search_forest.best_estimator_.score(X_test, Y_test)))


# In[64]:

forest_final = grid_search_forest.best_estimator_
feature_imprtance = list(zip(data_new.columns, forest_final.feature_importances_))
dtype = [('feature', 'S10'), ('importance', 'float')]
feature_imprtance = np.array(feature_imprtance, dtype = dtype)
feature_sort = np.sort(feature_imprtance, order='importance')[::-1]
feature_sort[0:]


# In[65]:

#Gradient Boosting 


# In[66]:

from sklearn.ensemble import GradientBoostingClassifier
gbm = GradientBoostingClassifier(max_features=10, n_estimators=200, 
                                 learning_rate=0.05, random_state= 2015)
gbm.fit(X_train, Y_train)


# In[67]:

import sklearn.metrics
print(sklearn.metrics.roc_auc_score(Y_train, gbm.predict_proba(X_train)[:,1]))
print(sklearn.metrics.roc_auc_score(Y_test, gbm.predict_proba(X_test)[:,1]))


# In[68]:

def plot_gbt_learning(gbt):
    test_score = np.empty(len(gbt.estimators_))
    train_score = np.empty(len(gbt.estimators_))
    for i, pred in enumerate(gbt.staged_predict_proba(X_test)):
         test_score[i] = sklearn.metrics.roc_auc_score(Y_test, pred[:,1])
    for i, pred in enumerate(gbt.staged_predict_proba(X_train)):
         train_score[i] = sklearn.metrics.roc_auc_score(Y_train, pred[:,1])
    plt.figure(figsize=(8,6))
    plt.plot(np.arange(gbt.n_estimators) + 1, test_score, label='Test') 
    plt.plot(np.arange(gbt.n_estimators) + 1, train_score, label='Train')
    plt.ylim(0,1.1)
plot_gbt_learning(gbm)


# In[69]:

feature_importance = gbm.feature_importances_ 
feature_importance = 100.0 * (feature_importance / feature_importance.max())
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.barh(pos[-30:], feature_importance[sorted_idx][-30:], align='center')
plt.yticks(pos[-30:], X_train.columns[[sorted_idx]][-30:])
plt.xlabel('Relative Importance')
plt.title('Variable Importance')
plt.figure(figsize=(20,20))
plt.show()


# In[70]:

len(data_new.columns)


# In[71]:

len(forest_final.feature_importances_)


# In[72]:

data_new.columns


# In[73]:

a = forest_final.feature_importances_
b = data_new.columns.values
#c = dstack((a,b))

feature_imprtancee = list(zip(b, a))
dtype = [('feature', 'S10'), ('importance', 'float')]
feature_imprtancee = np.array(feature_imprtancee, dtype = dtype)
feature_sort = np.sort(feature_imprtancee, order='importance')[::-1]
feature_sort[0:15]


# In[74]:

#data_new.UnkNameMatch.value_counts()


# In[ ]:

#Categorical:
#Age, Name Match, Match, Phone_State_Match


# In[ ]:

#Age_bcket.head(3)


# In[ ]:

#name_match.head(3)


# In[ ]:

#Continuous:
#numbers, stopz, S_x, Smiss, DomainEnd_Sx, Smiss_DE
#Numerical:
#Email_Total_Length, LP_length, domain_vowel_count
#name_match


# In[ ]:

#may or may not want to keep it wide to include domain aspects to see which one is more feature important


# In[ ]:

test_df.dtypes


# In[ ]:




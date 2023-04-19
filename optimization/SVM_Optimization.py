#!/usr/bin/env python
# coding: utf-8

# In[38]:


import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import NuSVC
from sklearn.svm import SVR
from sklearn.utils import shuffle
import random
from sklearn.preprocessing import LabelEncoder,OneHotEncoder


# In[39]:


df3 = pd.read_csv('C:/Users/Admin/Downloads/optimization/contentDataPrime.csv')


# In[40]:


df3.head()


# In[41]:


df3.isna().sum()


# In[42]:


df_dup3 = df3.duplicated('dataId', keep = False)
df_dup3.value_counts()


# In[43]:


df3 = df3.drop_duplicates(subset ='dataId')


# In[44]:


df3.replace(' ?', np.nan, inplace=True)


# In[45]:


df3.dropna(inplace=True)
df3.isna().sum()





# In[53]:


categ = ['contentType','title','certificate','description','releaseYear','endYear','gross','votes','rating','length']

# Encode Categorical Columns
le = LabelEncoder()
df3[categ] = df3[categ].apply(le.fit_transform)


# In[54]:


for i in range(10):
    sample = df3.sample(frac=0.40, replace=False)
    x = sample.iloc[:, :-1]
    y = sample.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=0)

    train_data = pd.concat([x_train, y_train], axis=1)
    test_data = pd.concat([x_test, y_test], axis=1)

    train_data.to_csv(f'sample_{i+1}_train.csv', index=False)
    test_data.to_csv(f'sample_{i+1}_test.csv', index=False)
    
    print(f"Sample {i+1}: Train={len(train_data)}, Test={len(test_data)}")


# In[48]:


kernels = ['linear', 'poly', 'rbf', 'sigmoid']
nus = [0]
epsilons = [0]
best_params = []
best_scores = []

results=[]


# In[ ]:


for i in range(10):
    train_data = pd.read_csv(f'sample_{i+1}_train.csv')
    test_data = pd.read_csv(f'sample_{i+1}_test.csv')
    x_train = train_data.drop('rating', axis=1)
    y_train = train_data['rating']
    x_test = test_data.drop('rating', axis=1)
    y_test = test_data['rating']

    best_score = 0
    best_param = {}
    for k in kernels:
        for j in range(1000):
            nu = random.uniform(0, 1)
            eps = random.uniform(0, 1)
            svm = NuSVC(kernel=k, nu=nu, gamma='auto')
            svm = SVR(kernel=k, epsilon=eps, C=1.0)
            svm.fit(x_train, y_train)
            score = svm.score(x_test, y_test)
            if score > best_score:
                best_score = score
                best_param = {'kernel': k, 'nu': nu, 'epsilon': eps}
    best_scores.append(best_score)
    best_params.append(best_param)
    results.append({'accuracy': accuracy, 'kernel': best_params['kernel'], 'nu': best_params['nu'], 'epsilon': best_params['epsilon']})


# In[ ]:


results_df = pd.DataFrame(results)
results_df.index.name = 'Sample'
display(results_df)


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/7_logistic_reg/insurance_data.csv')


# In[3]:


df.head()


# In[4]:


plt.scatter(df.age,df.bought_insurance,marker='*')


# In[5]:


df.shape


# In[6]:


from sklearn.model_selection import train_test_split


# In[8]:


x_train,x_test,y_train,y_test=train_test_split(df[['age']],df.bought_insurance,test_size=0.1)


# In[9]:


x_train


# In[10]:


x_test


# In[11]:


y_train


# In[12]:


y_test


# In[13]:


from sklearn.linear_model import LogisticRegression


# In[14]:


model = LogisticRegression()


# In[15]:


model.fit(x_train,y_train)


# In[16]:


model.predict(x_test)


# In[17]:


y_test


# In[19]:


model.score(x_test,y_test)


# In[20]:


model.predict_proba(x_test)


# In[ ]:





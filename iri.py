
# coding: utf-8

# In[1]:


#importing libraries 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
warnings.filterwarnings('ignore')

import seaborn as sns
import numpy as np
from sklearn import preprocessing


# In[2]:


#reading csv file
data = pd.read_csv('../Desktop/IRIS.csv')


# In[3]:


#prints first five rows
data.head()


# In[4]:


#gives information of data
data.info()


# In[5]:


#gives information like mean, count etc.
data.describe()


# In[6]:


#counts unique values in a  column
data['Species'].value_counts()


# In[7]:



plt_1=data.copy() #copying data 
for c in data.columns[data.dtypes == 'object']: #label encoder
    plt_1[c] = plt_1[c].factorize()[0]


# In[8]:


plt_1.head()


# In[9]:


# Correlation between numerical variables
"""plt_1 = data.copy()"""
matrix = plt_1.corr()
f, ax = plt.subplots(figsize=(9, 6))
sns.heatmap(matrix, vmax=.8, square=True, cmap="BuPu");


# In[10]:


plt_2=data.copy() 


# In[11]:


# Sepal Length vs Species
plt.figure(figsize=(12,4))
sns.barplot(plt_2['SepalLengthCm'], plt_2['Species'])


# In[12]:


"""Similarly visualize for other features"""


# In[13]:


#showing relation between features according to species
tmp = data.drop('Id', axis=1)
g = sns.pairplot(tmp, hue='Species', markers='+')
plt.show()


# In[14]:


#importing linear regression
from sklearn.linear_model import LinearRegression


# In[15]:


X=data.copy()
for c in data.columns[data.dtypes == 'object']:
    X[c] = X[c].factorize()[0]


# In[16]:


#copying species as target variable
y=X.Species
#dropping species from X
X=X.drop('Species', axis=1)


# In[17]:


#training linear regression model 
lr= LinearRegression()
model= lr.fit(X, y, sample_weight=None)
#Evaluating trained model
model.score(X, y, sample_weight=None)


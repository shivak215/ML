
# coding: utf-8

# In[3]:


from sklearn.datasets import load_digits
digits = load_digits()


# In[6]:


# Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
print("Image Data Shape" , digits.data.shape)
# Print to show there are 1797 labels (integers from 0–9)
print("Label Data Shape", digits.target.shape)
digits.data[0]


# In[6]:


import numpy as np 
import matplotlib.pyplot as plt
plt.figure(figsize=(20,4))
for index, (image, label) in enumerate(zip(digits.data[0:5], digits.target[0:5])):
     plt.subplot(1, 5, index + 1)
     plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
     plt.title('Training: %i\n' % label, fontsize = 20)


# In[7]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.30, random_state=0)


# In[8]:


from sklearn.linear_model import LogisticRegression


# In[9]:


# all parameters not specified are set to their defaults
logisticRegr = LogisticRegression()


# In[10]:


logisticRegr.fit(x_train, y_train)


# In[11]:


predictions = logisticRegr.predict(x_test)


# In[12]:


# Use score method to get accuracy of model
score = logisticRegr.score(x_test, y_test)
print(score)


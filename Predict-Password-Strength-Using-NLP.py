#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


data = pd.read_csv("D:\PyCharm Community Edition 2021.3.1\python projects\predict_password_Strength.csv", error_bad_lines = False)
data.head()


# In[3]:


data['strength'].unique()


# In[4]:


data.isna().sum()


# In[5]:


data[data['password'].isna()]


# In[6]:


data.dropna(inplace = True)


# In[7]:


data.isna().sum()


# In[8]:


sns.countplot(data['strength'])


# In[9]:


password_tuple = np.array(data)
password_tuple


# In[10]:


import random
random.shuffle(password_tuple)


# In[11]:


x = [labels[0] for labels in password_tuple]
y = [labels[1] for labels in password_tuple]


# In[12]:


x


# In[13]:


def word_divide_char(inputs):
     return list(inputs)


# In[14]:


word_divide_char('kzde5577')


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer=TfidfVectorizer(tokenizer=word_divide_char)
X = vectorizer.fit_transform(x)


# In[16]:


X.shape


# In[17]:


vectorizer.get_feature_names()


# In[18]:


first_document_vector = X[0]
first_document_vector


# In[19]:


first_document_vector.T.todense()


# In[20]:


df = pd.DataFrame(first_document_vector.T.todense(),vectorizer.get_feature_names(), columns = ['TF-IDF'])


# In[21]:


df.sort_values(by=['TF-IDF'],ascending=False)


# In[22]:


from sklearn.model_selection import train_test_split


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# In[24]:


X_train.shape


# In[25]:


from sklearn.linear_model import LogisticRegression


# In[26]:


clf = LogisticRegression(random_state=42, multi_class='multinomial')


# In[27]:


clf.fit(X_train, y_train)


# In[28]:


# doing prediction for specific custom data

dt=np.array(['%@123abcd'])
pred=vectorizer.transform(dt)
clf.predict(pred)


# In[29]:


# doing prediction on X-Test data
y_pred=clf.predict(X_test)
y_pred


# In[30]:


# check Accuracy of your model using confusion_matrix,accuracy_score
from sklearn.metrics import confusion_matrix,accuracy_score
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(accuracy_score(y_test,y_pred))


# In[31]:


# create report of your model
from sklearn.metrics import classification_report
print(classification_report(y_test,y_pred))


# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


#decision tree classifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split 
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[10]:


iris = pd.read_csv(r'D:\boook\iris.csv')


# In[11]:


iris.head()   


# In[12]:


iris.tail()


# In[13]:


iris.describe()


# In[14]:


iris.shape


# In[7]:


iris.info()


# In[15]:


iris.isnull().sum()


# In[16]:


iris.value_counts()


# In[17]:


iris.isnull().any()


# In[18]:


iris.columns


# In[19]:


iris.dtypes


# In[20]:


iris.corr()


# In[21]:


sns.pairplot(iris.iloc[:,1:])


# In[22]:


iris.hist(edgecolor="black", linewidth=0.75)


# In[23]:


sns.heatmap(iris.corr(),annot=True)


# In[24]:


iris.dtypes


# In[25]:


feature_cols=['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
x=iris[feature_cols]
y=iris.species


# In[26]:


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=1)


# In[27]:


#building model
dtc=DecisionTreeClassifier()
dtc=dtc.fit(x_train, y_train)
y_pred = dtc.predict(x_test)
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
result = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(result)
result1 = classification_report(y_test, y_pred)
print("Classification Report:",)
print (result1)
result2 = accuracy_score(y_test,y_pred)
print("Accuracy:",result2)


# In[28]:


#plotting decision tree
features = iris.columns[:-1]
classes = iris['species'].unique().tolist()

from sklearn.tree import plot_tree
plt.figure(figsize=(19, 14))
plot_tree(dtc, feature_names=features, class_names=classes, filled=True)
plt.show()


# In[ ]:





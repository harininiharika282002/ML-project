#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


data_set=pd.read_csv(r'C:\Users\91822\Downloads\terror1.csv')


# In[5]:


data_set.columns.to_list() 


# In[6]:


data_set.rename(columns={'iyear':'Year','imonth':'Month','city':'City','iday':'Day','country_txt':'Country','region_txt':'Region','attacktype1_txt':'AttackType','target1':'Target','nkill':'Killed','nwound':'Wounded','summary':'Summary','gname':'Group','targtype1_txt':'Target_type','weaptype1_txt':'Weapon_type','motive':'Motive'},inplace=True)
data_set['Casualities'] = data_set.Killed + data_set.Wounded
data_set=data_set[['Year','Month','Day','Country','Region','City','latitude','longitude','AttackType','Killed','Wounded','Casualities','Target','Group','Target_type','Weapon_type']]
data_set.columns 


# In[7]:


data_set['Casualities'] = data_set.Killed + data_set.Wounded
data_set.head()


# In[8]:


data_set.tail()


# In[9]:


data_set.shape


# In[10]:


data_set.isnull().sum()


# In[11]:


data_set.info()


# In[12]:


data_set.describe(include='all')


# In[13]:


print('Country with most attacks: ',data_set['Country'].value_counts().idxmax())
print('City with most attacks: ',data_set['City'].value_counts().index[1])
print("Region with the most attacks:",data_set['Region'].value_counts().idxmax())
print("Year with the most attacks:",data_set['Year'].value_counts().idxmax())
print("Month with the most attacks:",data_set['Month'].value_counts().idxmax())
print("Group with the most attacks:",data_set['Group'].value_counts().index[1])
print("Most Attack Types:",data_set['AttackType'].value_counts().idxmax())


# In[14]:


pd.DataFrame(data_set['Year'].value_counts(dropna=False).sort_index())


# In[16]:


most_terror = data_set['Country'].value_counts()
pd.DataFrame(most_terror.head(5))


# In[17]:


pd.DataFrame(data_set['City'].value_counts()[:5])


# In[18]:


x_year = data_set['Year'].unique()
y_year = data_set['Year'].value_counts(dropna=False).sort_index()
plt.figure(figsize=(15,10))
sns.barplot(x=x_year,y=y_year, palette= 'flare')
plt.xticks(rotation=45)
plt.title("Attack in Years")
plt.xlabel('Years')
plt.ylabel('Number of attacks each year')
plt.show()


# In[19]:


attack_country = data_set.Country.value_counts()[:15]
pd.DataFrame(attack_country)


# In[20]:


plt.subplots(figsize=(8,4))
sns.barplot(attack_country.index,attack_country.values,palette='Blues_d')
plt.title('Top Countries Affected')
plt.xlabel('Countries')
plt.ylabel('Count')
plt.xticks(rotation= 45)
plt.show()


# In[21]:


chart=sns.catplot(x='Weapon_type',y='Year',kind='box',height=5,aspect=3,data=data_set,orient="v")
chart.set_xticklabels(rotation=90)

chart.set_yticklabels(rotation=90)


# In[ ]:





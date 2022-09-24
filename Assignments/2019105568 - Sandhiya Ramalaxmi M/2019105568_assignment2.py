#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


d=pd.read_csv("Churn_Modelling.csv")


# In[4]:


d.columns


# Univariate analysis

# In[7]:


sns.boxplot(d['CreditScore'])


# In[8]:


plt.hist(d['Age'])
plt.show()


# In[9]:


sns.barplot(d['Gender'], d['Age'])


# In[10]:


pie_chart=plt.pie(d['Age'].head(), autopct="%.2f")
plt.show(pie_chart)


# In[11]:


sns.distplot(data['Age'].head(200))


# BIVARIATE ANALYSIS

# In[12]:


plt.scatter(d['CreditScore'].head(100),d['Age'].head(100))
plt.show()


# In[13]:


plt.bar(data['RowNumber'].head() ,data['CreditScore'].head(),  )

plt.title('bar_Plot')
plt.xlabel('row_number')
plt.ylabel('credit_Score')


# In[14]:


sns.jointplot(d['RowNumber'].head(100) ,d['CreditScore'].head(100),  )


# MULTIVARIATE ANALYSIS

# In[15]:


sns.barplot('RowNumber','Age',hue='Geography', d=d.head())


# In[16]:


fig= plt.figure(figsize  =(12,10))
sns.heatmap(d.head().corr(), annot = True)


# In[19]:


fig= plt.figure(figsize  =(7,5))
sns.pairplot(d.head(), hue='EstimatedSalary')


# In[ ]:





# Descriptive statistics

# In[20]:


d.head()


# In[21]:


d.info()


# In[22]:


d.describe()


# Handling of missing values

# In[24]:


d.isna().sum()


# Checking for outliers and replacing them

# In[25]:


sns.boxplot(d['Age'])


# In[26]:


qt= d.quantile(q=[0.25,0.75])
qt


# In[27]:


irq=qt.loc[0.75]- qt.loc[0.25] # q3 and q1
irq


# In[28]:


upper= qt.loc[0.75]+(1.5*irq)
upper


# In[29]:


d['Age'].mean()


# Categorical data and Encoding

# In[30]:


d.Geography.unique()


# In[31]:


d['Gender'].replace({'Female':0, 'Male': 1 }, inplace=True)
d['Geography'].replace({'France':0,'Germany':1, 'Spain':2}, inplace=True)
d.head()


# In[32]:


# using dummy values
data_d= pd.get_dummies(d,columns = ['Surname'])
data_d.head()


# In[ ]:





# Splitting the data into dependent and independent variables
# 

# In[33]:


x=data_d.drop(columns= ['EstimatedSalary']).values
y=data_d['EstimatedSalary'].values
print(x)
print(y)


# Scaling the independent variables

# In[34]:


from sklearn.preprocessing import scale


# In[35]:


x = scale(x)
x


# Splitting the data into training and testing

# In[36]:


from sklearn.model_selection import train_test_split


# In[37]:


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)


# In[38]:


print(x_train.shape, x_test.shape)


# In[ ]:





# In[ ]:





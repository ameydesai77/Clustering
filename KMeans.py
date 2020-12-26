#!/usr/bin/env python
# coding: utf-8
The Mall customers dataset holds the details about people visiting the mall. The dataset has an age, customer id, gender, annual income, and spending score. It gains insights from the data and divides the customers into different groups based on their behaviors.
# #### Importing the libraries

# In[1]:


import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd


# #### Importing the dataset

# In[2]:


dataset = pd.read_csv('....\DATASETS\mall_customers.csv')


# ##### since  k_Means is unsupervised machine learning model we consider only feature columns 

# In[3]:


X = dataset.iloc[:, [3, 4]].values
# y = dataset.iloc[:, 3].values


# In[4]:


plt.scatter(dataset['Annual Income (k$)'],dataset['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')


# In[5]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(dataset[['Annual Income (k$)','Spending Score (1-100)']])
y_predicted


# In[6]:


dataset['cluster']=y_predicted
dataset.head()


# #### Splitting the dataset into the Training set and Test set
# 
# #### The dataset contains very less number of rows ,thus we will consider full dataset instead of split

# In[ ]:





# In[7]:


km = KMeans(n_clusters=3)
y_predicted = km.fit_predict(dataset[['Annual Income (k$)','Spending Score (1-100)']])
y_predicted


# In[8]:


df1 = dataset[dataset.cluster==0]
df2 = dataset[dataset.cluster==1]
df3 = dataset[dataset.cluster==2]
plt.scatter(df1['Annual Income (k$)'],df1['Spending Score (1-100)'],color='cyan')
plt.scatter(df2['Annual Income (k$)'],df2['Spending Score (1-100)'],color='gray')
plt.scatter(df3['Annual Income (k$)'],df3['Spending Score (1-100)'],color='yellow')
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='blue',marker='*',label='centroid')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()


# #### Preprocessing using min max scaler

# In[9]:


scaler = MinMaxScaler()
dataset['Annual Income (k$)'] = scaler.fit_transform(dataset[['Annual Income (k$)']])
dataset['Spending Score (1-100)'] = scaler.fit_transform(dataset[['Spending Score (1-100)']])


# In[10]:


dataset.head()


# In[11]:


plt.scatter(dataset['Annual Income (k$)'],dataset['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')


# In[12]:


km = KMeans(n_clusters=5)
y_predicted = km.fit_predict(dataset[['Annual Income (k$)','Spending Score (1-100)']])


# In[13]:


km.cluster_centers_


# In[14]:


dataset['cluster']=y_predicted
dataset.head()


# #### Using the Elbow method to find the optimal number of clusters

# In[15]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ### Fitting K-Means to the dataset

# In[16]:


kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(X)


# #### Visualising the clusters

# In[17]:


df1 = dataset[dataset.cluster==0]
df2 = dataset[dataset.cluster==1]
df3 = dataset[dataset.cluster==2]
df4 = dataset[dataset.cluster==3]
df5 = dataset[dataset.cluster==4]
plt.scatter(df1['Annual Income (k$)'],df1['Spending Score (1-100)'],color='cyan')
plt.scatter(df2['Annual Income (k$)'],df2['Spending Score (1-100)'],color='gray')
plt.scatter(df3['Annual Income (k$)'],df3['Spending Score (1-100)'],color='yellow')
plt.scatter(df4['Annual Income (k$)'],df4['Spending Score (1-100)'],color='green')
plt.scatter(df5['Annual Income (k$)'],df5['Spending Score (1-100)'],color='pink')


plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='blue',marker='*',label='centroid')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()


# In[ ]:





#!/usr/bin/env python
# coding: utf-8
Objective of applying Hierarichal technique is to identify the number of clusters to be formed for the smooth analysis of the data 
# ### Import required libraries

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# #### Importing the dataset
# 

# In[21]:


dataset=pd.read_csv('...\DATASETS\mall_Customers.csv')


# In[26]:


dataset.head()


# In[27]:


dataset.columns


# ### Select the feature columns

# In[8]:


x=dataset.iloc[:,[3,4]].values


# ### Using the dendrogram to find the optimal number of clusters
# 

# In[9]:


import scipy.cluster.hierarchy as sch


# In[10]:



dendogram=sch.dendrogram(sch.linkage(x,method='ward'))
plt.title('Dendogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean Distance')
plt.show()


# ### Obtaining the distance from cluster to cluster

# In[11]:


from scipy.spatial.distance import cdist


# In[12]:


ds = cdist(x,x)


# In[13]:


ds


# In[14]:


from scipy.cluster.hierarchy import *
lm = linkage(ds,"single")
lm[:50]


# ### Visualising the clusters

# In[15]:



plt.scatter(dataset['Annual Income (k$)'],dataset['Spending Score (1-100)'])
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')


# ### Fitting Hierarchical Clustering to the dataset
# 

# In[16]:


from sklearn.cluster import AgglomerativeClustering


# In[17]:


hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')


# In[18]:


y_hc=hc.fit_predict(x)


# ### Visualising the clusters
# 

# In[19]:


plt.scatter(x[y_hc == 0,0],x[y_hc == 0,1],s=100,c='red',label='cluster_1')
plt.scatter(x[y_hc == 1,0],x[y_hc == 1,1],s=100,c='green',label='cluster_2')
plt.scatter(x[y_hc == 2,0],x[y_hc == 2,1],s=100,c='blue',label='cluster_3')
plt.scatter(x[y_hc == 3,0],x[y_hc == 3,1],s=100,c='yellow',label='cluster_4')
plt.scatter(x[y_hc == 4,0],x[y_hc == 4,1],s=100,c='cyan',label='cluster_5')
plt.title('customers of Customers')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score(1-100)')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as hcluster


# In[5]:


# Import data
sensor_geom = pd.read_csv('../sensor_geometry.csv')


# In[8]:


batch1 = pd.read_parquet('../batches_train/batch_1_repartitions/part.0.parquet')


# In[16]:


np.unique(b['sensor_id'].values)


# In[6]:


# Make a function that outputs (x,y,z) for a sensor_id input
def id_to_xyz(sen):
    row = tuple(sensor_geom.loc[sen][1:4])
    return row


# In[34]:


# Given an event, find out how many clusters it has
def how_many_clusters(event, aux_incl=False, threshold=150, criterion='distance'):
    if aux_incl == False:
        event = event[event.auxiliary==False]
    raw_data = [id_to_xyz(sen) for sen in np.unique(event['sensor_id'].values)]
    clusters = hcluster.fclusterdata(raw_data,threshold,criterion=criterion)
    return np.unique(clusters).size


# In[37]:


how_many_clusters(batch1.loc[24],False)


# In[ ]:





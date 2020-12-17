#!/usr/bin/env python
# coding: utf-8

# In[22]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import seaborn as sn
import os
print(os.listdir("C:/Users/Vikas/Downloads/Compressed/NASAData"))

# Any results you write to the current directory are saved as output.


# In[2]:


columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 'sen1', 'sen2', 'sen3','sen4', 'sen5', 'sen6', 'sen7', 'sen8',
         'sen9', 'sen10', 'sen11', 'sen12', 'sen13', 'sen14', 'sen15', 'sen16', 'sen17', 'sen18', 'sen19', 'sen20', 'sen21']

feature_columns = ['setting1', 'setting2', 'setting3', 'sen1', 'sen2', 'sen3','sen4', 'sen5', 'sen6', 'sen7', 'sen8',
         'sen9', 'sen10', 'sen11', 'sen12', 'sen13', 'sen14', 'sen15', 'sen16', 'sen17', 'sen18', 'sen19', 'sen20', 'sen21', 'cycle_norm']


# In[3]:


fd_number = '1'


# In[4]:


train_df = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/train_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
test_df = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/test_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
rul_df = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/RUL_FD00" + fd_number + ".txt", delimiter="\s+", header=None)


# In[5]:


fd_number = '2'
train_df2 = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/train_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
test_df2 = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/test_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
rul_df2 = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/RUL_FD00" + fd_number + ".txt", delimiter="\s+", header=None)


# In[6]:


fd_number = '3'
train_df3 = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/train_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
test_df3 = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/test_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
rul_df3 = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/RUL_FD00" + fd_number + ".txt", delimiter="\s+", header=None)


# In[7]:


fd_number = '4'
train_df4 = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/train_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
test_df4 = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/test_FD00" + fd_number + ".txt", delimiter="\s+", header=None)
rul_df4 = pd.read_csv("C:/Users/vikas/Downloads/Compressed/NASAData/RUL_FD00" + fd_number + ".txt", delimiter="\s+", header=None)


# In[8]:


train_df.columns = columns
test_df.columns = columns
rul_df.columns = ['truth']
rul_df['id'] = rul_df.index + 1
rul_df['dataset_id'] = 'FD001'
test_df['dataset_id'] = 'FD001'
train_df['dataset_id'] = 'FD001'


# In[10]:


train_df2.columns = columns
test_df2.columns = columns
rul_df2.columns = ['truth']
rul_df2['id'] = rul_df2.index + 1
rul_df2['dataset_id'] = 'FD002'
test_df2['dataset_id'] = 'FD002'
train_df2['dataset_id'] = 'FD002'


# In[11]:


train_df3.columns = columns
test_df3.columns = columns
rul_df3.columns = ['truth']
rul_df3['id'] = rul_df3.index + 1
rul_df3['dataset_id'] = 'FD003'
test_df3['dataset_id'] = 'FD003'
train_df3['dataset_id'] = 'FD003'


# In[12]:


train_df4.columns = columns
test_df4.columns = columns
rul_df4.columns = ['truth']
rul_df4['id'] = rul_df4.index + 1
rul_df4['dataset_id'] = 'FD004'
test_df4['dataset_id'] = 'FD004'
train_df4['dataset_id'] = 'FD004'


# In[13]:


train_frames = [train_df, train_df2, train_df3 ,train_df4]
train_result = pd.concat(train_frames)
test_frames = [test_df, test_df2, test_df3 ,test_df4]
test_result = pd.concat(test_frames)
rul_frames = [rul_df, rul_df2, rul_df3 ,rul_df4]
rul_result = pd.concat(rul_frames)


# In[14]:


train_result.head()


# In[15]:


test_result.head()


# In[16]:


rul_result.head()


# In[21]:


rul_result['dataset_id'].hist()


# In[23]:


corrMatrix = rul_result.corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()


# In[27]:


train_result.hist(figsize=(20,15))


# In[36]:


pd.plotting.scatter_matrix(rul_result, alpha=0.2)


# In[28]:


sensor_columns = [col for col in train_df.columns if col.startswith("sen")]
setting_columns = [col for col in train_df.columns if col.startswith("setting")]
print(sensor_columns)
print(setting_columns)


# In[29]:


example_slice = train_df[(train_df.dataset_id == 'FD001') & (train_df.id == 1)]

fig, axes = plt.subplots(7, 3, figsize=(15, 10), sharex=True)
index = 0
for index, ax in enumerate(axes.ravel()):
    sensor_col = sensor_columns[index]
    example_slice.plot(x='cycle',y=sensor_col, ax=ax, color='Blue')
    
    if index % 3 == 0:
        ax.set_ylabel("Sensor Value", size=10)
    else:
        ax.set_ylabel("")
    
    ax.set_xlabel("Time (Cycles)")
   
    
fig.suptitle("Sensor Traces : Unit 1, Dataset 1", size=20, y=1.025)
fig.tight_layout()


# In[30]:


all_units = train_df[train_df['dataset_id'] == 'FD001']['id'].unique()
units_to_plot = np.random.choice(all_units, size=10, replace=False)
plot_data = train_df[(train_df['dataset_id'] == 'FD001') & (train_df['id'].isin(units_to_plot))].copy()
plot_data.head()


# In[31]:


for index, ax in enumerate(axes.ravel()):
    sensor_col = sensor_columns[index]
    for unit_id, group in plot_data.groupby('id'):
        c = group.drop(columns=['dataset_id'],axis=1)


# In[32]:


fig, axes = plt.subplots(7, 3, figsize=(15, 10), sharex=True)
for index, ax in enumerate(axes.ravel()):
    sensor_col = sensor_columns[index]
    for unit_id, group in plot_data.groupby('id'):
        temp = group.drop(['dataset_id'],axis=1)
        (temp.plot(x='cycle', y=sensor_col, alpha=0.45, ax=ax, color='gray', legend=False))
        (temp.rolling(window=10, on='cycle').mean().plot(x='cycle', y=sensor_col, alpha=.75, ax=ax, color='blue', legend=False));
    if index % 3 == 0:
        ax.set_ylabel('Sensor Value', size=10)
    else:
        ax.set_ylabel('')
    
    ax.set_xlabel('Time (Cycles)')
fig.suptitle('All Sensor Traces: Dataset 1 (Random Sample of 10 Units)', size=20, y=1.025)
fig.tight_layout()


# In[33]:


def cycles_until_failure(r, lifetimes):
    return r['cycle'] - lifetimes.ix[(r['dataset_id'], r['id'])]


# In[34]:


lifetimes = train_df.groupby(['dataset_id','id'])['cycle'].max()
plot_data['ctf'] = plot_data.apply(lambda r: cycles_until_failure(r, lifetimes), axis=1)

fig, axes = plt.subplots(7,3, figsize=(15,10), sharex = True)
for index, ax in enumerate(axes.ravel()):
    sensor_col = sensor_columns[index]
    for unit_id, group in plot_data.groupby('id'):
        temp = group.drop(['dataset_id'],axis=1)
        (temp.plot(x='ctf', y=sensor_col, alpha=0.45, ax=ax, color='gray', legend=False))
        (temp.rolling(window=10,on='ctf').mean().plot(x='ctf',y=sensor_col, alpha=.75, ax=ax, color='black',legend=False))
    if index % 3 == 0:
        ax.set_ylabel("Sensor Value", size=10)
    else:
        ax.set_ylabel("")
    ax.set_title(sensor_col.title())
    ax.set_xlabel('Time Before Failure (Cycles)')
    ax.axvline(x=0, color='r', linewidth=3)
    ax.set_xlim([None,10])
fig.suptitle("All Sensor Traces: Dataset 1 (Random Sample of 10 Units)", size=20, y=1.025)
fig.tight_layout()


# In[ ]:





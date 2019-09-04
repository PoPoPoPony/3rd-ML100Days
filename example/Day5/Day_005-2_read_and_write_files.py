#!/usr/bin/env python
# coding: utf-8

# ## 使用內建功能讀取 txt 檔

# # [教學目標]
# - 示範 Pandas 各種 讀取 / 寫入 檔案的方式

# # [範例重點]
# - 讀取 txt 檔 (In[2], Out[2])
# - 存取 json 檔 (In[4], In[5], In[7], In[8])
# - 存取 npy 檔 (numpy專用檔, In[10], In[11]) 
# - 讀取 Pickle 檔 (In[12], In[13])

# In[1]:


with open("data/example.txt", 'r') as f:
    data = f.readlines()
print(data)


# ## 將 txt 轉成 pandas dataframe

# In[2]:


import pandas as pd

data = []
with open("data/example.txt", 'r') as f:
    for line in f:
        line = line.replace('\n', '').split(',') # 將每句最後的 /n 取代成空值後，再以逗號斷句
        data.append(line)
data


# In[3]:


df = pd.DataFrame(data[1:])
df.columns = data[0]
df


# ## 將資料轉成 json 檔後輸出
# 將 json 讀回來後，是否與我們原本想要存入的方式一樣? (以 id 為 key)

# In[4]:


import json
df.to_json('data/example01.json')


# In[5]:


# 上面的存入方式，會將 column name 做為主要的 key, row name 做為次要的 key
with open('data/example01.json', 'r') as f:
    j1 = json.load(f)
j1


# In[6]:


df.set_index('id', inplace=True)
df


# In[7]:


df.to_json('data/example02.json', orient='index')


# In[8]:


with open('data/example02.json', 'r') as f:
    j2 = json.load(f)
j2


# ## 將檔案存為 npy 檔
# 一個專門儲存 numpy array 的檔案格式
# 使用 npy 通常可以讓你更快讀取資料喔!  
# [建議閱讀](https://towardsdatascience.com/why-you-should-start-using-npy-file-more-often-df2a13cc0161)

# In[9]:


import numpy as np
# 將 data 的數值部分轉成 numpy array
array = np.array(data[1:])
array


# In[10]:


np.save(arr=array, file='data/example.npy')


# In[11]:


array_back = np.load('data/example.npy')
array_back


# ## Pickle
# 存成 pickle 檔  
# 什麼都包，什麼都不奇怪的 [Pickle](https://docs.python.org/3/library/pickle.html)  
# 比如說 [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) 的資料集就是用 pickle 包的喔!

# In[12]:


import pickle
with open('data/example.pkl', 'wb') as f:
    pickle.dump(file=f, obj=data)


# In[13]:


with open('data/example.pkl', 'rb') as f:
    pkl_data = pickle.load(f)
pkl_data


# In[ ]:





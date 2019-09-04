#!/usr/bin/env python
# coding: utf-8

# # [作業目標]
# - 仿造範例的 One Hot Encoding, 將指定的資料進行編碼

# # [作業重點]
# - 將 sub_train 進行 One Hot Encoding 編碼 (In[4], Out[4])

# In[1]:


import os
import numpy as np
import pandas as pd


# In[2]:


# 設定 data_path, 並讀取 app_train
dir_data = 'D:/VScode workshop/3rd-ML100Days/practice/Day6/data/'
f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)


# ## 作業
# 將下列部分資料片段 sub_train 使用 One Hot encoding, 並觀察轉換前後的欄位數量 (使用 shape) 與欄位名稱 (使用 head) 變化

# In[3]:


sub_train = pd.DataFrame(app_train['WEEKDAY_APPR_PROCESS_START'])
print(sub_train.shape)
sub_train.head()


# In[ ]:

one_hot_train = pd.get_dummies(sub_train)
print(one_hot_train.shape)
print(one_hot_train.head())



"""
Your Code Here
"""


#!/usr/bin/env python
# coding: utf-8

# ### 讀取資料
# 首先，我們用 pandas 讀取最主要的資料 application_train.csv (記得到 https://www.kaggle.com/c/home-credit-default-risk/data 下載)
# 
# Note: `data/application_train.csv` 表示 `application_train.csv` 與該 `.ipynb` 的資料夾結構關係如下
# ```
# data
#     /application_train.csv
# Day_004_first_EDA.ipynb
# ```

# # [教學目標]
# - 初步熟悉以 Python 為主的資料讀取與簡單操作

# # [範例重點]
# - 如何使用 pandas.read_csv 讀取資料 (In[3], Out[3])
# - 如何簡單瀏覽 pandas 所讀進的資料 (In[5], Out[5])

# In[1]:


import os
import numpy as np
import pandas as pd


# In[2]:


# 設定 data_path
dir_data = './data/'


# #### 用 pd.read_csv 來讀取資料

# In[3]:


f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)


# #### Note: 在 jupyter notebook 中，可以使用 `?` 來調查函數的定義

# In[4]:


# for example
get_ipython().run_line_magic('pinfo', 'pd.read_csv')


# #### 接下來我們可以用 .head() 這個函數來觀察前 5 row 資料

# In[5]:


app_train.head()


# ## 練習時間
# 資料的操作有很多，接下來的馬拉松中我們會介紹常被使用到的操作，參加者不妨先自行想像一下，第一次看到資料，我們一般會想知道什麼訊息？
# 
# #### Ex: 如何知道資料的 row 數以及 column 數、有什麼欄位、多少欄位、如何截取部分的資料等等
# 
# 有了對資料的好奇之後，我們又怎麼通過程式碼來達成我們的目的呢？
# 
# #### 可參考該[基礎教材](https://bookdata.readthedocs.io/en/latest/base/01_pandas.html#DataFrame-%E5%85%A5%E9%97%A8)或自行 google

# In[ ]:





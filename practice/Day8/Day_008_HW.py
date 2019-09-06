#!/usr/bin/env python
# coding: utf-8

# # [作業目標]
# - 對資料做更多處理 : 顯示特定欄位的統計值與直方圖

# # [作業重點]
# - 試著顯示特定欄位的基礎統計數值 (In[4], Out[4], Hint : describe())
# - 試著顯示特定欄位的直方圖 (In[5], Out[5], Hint : .hist())

# In[ ]:


# Import 需要的套件
import os
import numpy as np
import pandas as pd

# 設定 data_path
dir_data = 'D:/VScode workshop/3rd-ML100Days/practice/Day8/data/'


# In[ ]:


f_app_train = os.path.join(dir_data, 'application_train.csv')
app_train = pd.read_csv(f_app_train)


# In[ ]:


import matplotlib.pyplot as plt


# ## 練習時間

# 觀察有興趣的欄位的資料分佈，並嘗試找出有趣的訊息
# #### Eg
# - 計算任意欄位的平均數及標準差
# - 畫出任意欄位的[直方圖](https://zh.wikipedia.org/zh-tw/%E7%9B%B4%E6%96%B9%E5%9B%BE)
# 
# ### Hints:
# - [Descriptive Statistics For pandas Dataframe](https://chrisalbon.com/python/data_wrangling/pandas_dataframe_descriptive_stats/)
# - [pandas 中的繪圖函數](https://amaozhao.gitbooks.io/pandas-notebook/content/pandas%E4%B8%AD%E7%9A%84%E7%BB%98%E5%9B%BE%E5%87%BD%E6%95%B0.html)
# 

# In[ ]:

print(app_train.columns)
print(app_train.dtypes)

print(app_train['OWN_CAR_AGE'].mean())
print(app_train['OWN_CAR_AGE'].std())

print(app_train['OWN_CAR_AGE'].shape)

car_age_data = app_train['OWN_CAR_AGE'].value_counts().reset_index()
car_age_data.columns = ['age' , 'count']

print(car_age_data)


plt.hist(app_train['OWN_CAR_AGE'] , bins = range(0 , 60 , 10))
plt.title('histogram for OWN_CAR_AGE')
plt.show()


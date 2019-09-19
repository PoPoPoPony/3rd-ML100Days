#!/usr/bin/env python
# coding: utf-8

# ## 作業
# ### 請使用 application_train.csv, 根據不同的 HOUSETYPE_MODE 對 AMT_CREDIT 繪製 Histogram

# # [作業目標]
# - 試著調整資料, 並利用提供的程式繪製分布圖

# # [作業重點]
# - 如何將列出相異的 HOUSETYPE_MODE 類別 (In[3])
# - 如何依照不同的 HOUSETYPE_MODE 類別指定資料, 並繪製長條圖(.hist())? (In[3])

# In[1]:


# 載入需要的套件
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # 另一個繪圖-樣式套件

# 忽略警告訊息
plt.style.use('ggplot')
import warnings
warnings.filterwarnings('ignore')

# 設定 data_path
dir_data = 'D:/VScode workshop/3rd-ML100Days/practice/Day19/data/'


# In[2]:


# 讀取檔案
f_app = os.path.join(dir_data, 'application_train.csv')
print('Path of read in data: %s' % (f_app))
app_train = pd.read_csv(f_app)
app_train.head()


# In[ ]:


# 使用不同的 HOUSETYPE_MODE 類別繪製圖形, 並使用 subplot 排版
"""
Your Code Here
"""



unique_house_type = list(app_train['HOUSETYPE_MODE'].unique())
unique_house_type.pop(1)

nrows = len(unique_house_type)
ncols = nrows // 2

print(unique_house_type)


plt.figure(figsize=(10,30))
for i in range(len(unique_house_type)):
    plt.subplot(nrows, ncols, i+1)
    """
    Your Code Here
    """
    plt.hist(app_train.loc[app_train['HOUSETYPE_MODE'] == unique_house_type[i] , 'AMT_CREDIT'] , bins = 50)
    plt.title(str(unique_house_type[i]))
plt.show()



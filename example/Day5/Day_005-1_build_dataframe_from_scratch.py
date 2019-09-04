#!/usr/bin/env python
# coding: utf-8

# # [教學目標]
# - Pandas 處理最常用的資料格式, 我們稱為 DataFrame, 試著使用不同的方式新建一個 DataFrame 吧
# - 練習看看 DataFrame 可以對資料做什麼操作? (groupby 的使用)

# # [範例重點]
# - 新建 DataFrame 方法一 (In[2], In[3])
# - 新建 DataFrame 方法二 (In[4], In[5])
# - 資料操作 : groupby (In[6], Out[6])

# In[1]:


import pandas as pd


# ### 方法一

# In[2]:


data = {'weekday': ['Sun', 'Sun', 'Mon', 'Mon'],
        'city': ['Austin', 'Dallas', 'Austin', 'Dallas'],
        'visitor': [139, 237, 326, 456]}


# In[3]:


visitors_1 = pd.DataFrame(data)
print(visitors_1)


# ### 方法二

# In[4]:


cities = ['Austin', 'Dallas', 'Austin', 'Dallas']
weekdays = ['Sun', 'Sun', 'Mon', 'Mon']
visitors = [139, 237, 326, 456]

list_labels = ['city', 'weekday', 'visitor']
list_cols = [cities, weekdays, visitors]

zipped = list(zip(list_labels, list_cols))


# In[5]:


visitors_2 = pd.DataFrame(dict(zipped))
print(visitors_2)


# ## 一個簡單例子
# 假設你想知道如果利用 pandas 計算上述資料中，每個 weekday 的平均 visitor 數量，
# 
# 通過 google 你找到了 https://stackoverflow.com/questions/30482071/how-to-calculate-mean-values-grouped-on-another-column-in-pandas
# 
# 想要測試的時候就可以用 visitors_1 這個只有 4 筆資料的資料集來測試程式碼

# In[6]:


visitors_1.groupby(by="weekday")['visitor'].mean()


# ## 練習時間
# 在小量的資料上，我們用眼睛就可以看得出來程式碼是否有跑出我們理想中的結果
# 
# 請嘗試想像一個你需要的資料結構 (裡面的值可以是隨機的)，然後用上述的方法把它變成 pandas DataFrame
# 
# #### Ex: 想像一個 dataframe 有兩個欄位，一個是國家，一個是人口，求人口數最多的國家
# 
# ### Hints: [隨機產生數值](https://blog.csdn.net/christianashannon/article/details/78867204)

# In[ ]:





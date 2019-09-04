#!/usr/bin/env python
# coding: utf-8

# # [作業目標]
# - 利用範例的創建方式, 創建一組資料, 並練習如何取出最大值

# # [作業重點]
# - 練習創立 DataFrame (In[2])
# - 如何取出口數最多的國家 (In[3], Out[3])

# ## 練習時間
# 在小量的資料上，我們用眼睛就可以看得出來程式碼是否有跑出我們理想中的結果
# 
# 請嘗試想像一個你需要的資料結構 (裡面的值可以是隨機的)，然後用上述的方法把它變成 pandas DataFrame
# 
# #### Ex: 想像一個 dataframe 有兩個欄位，一個是國家，一個是人口，求人口數最多的國家
# 
# ### Hints: [隨機產生數值](https://blog.csdn.net/christianashannon/article/details/78867204)

# In[1]:


import pandas as pd
from numpy import random


# In[ ]:


population = random.randint(low = 100000 , high = 500000 , size = 5)
print(population)


data = {'國家': ['US' , 'UK' , 'JP' , 'PRC' , 'ROC'],
        '人口': population}
data = pd.DataFrame(data)


# In[ ]:

print(data)
print(data['國家'][data['人口'].idxmax()])



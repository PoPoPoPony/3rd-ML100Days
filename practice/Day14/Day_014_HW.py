#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# 作業
#參考範例程式碼，模擬一組負相關的資料，並計算出相關係數以及畫出 scatter plot


# In[ ]:


# [作業目標]
#以下程式碼將示範在 python 如何利用 numpy 計算出兩組數據之間的相關係數，並觀察散佈圖
#藉由觀察相關矩陣與散佈圖的關係, 希望同學對 負相關 的變數分布情形也有比較直覺的理解


# In[ ]:


# [作業重點]
#仿照 In[4], In[5] 的語法, 寫出負相關的變數, 並觀察相關矩陣以及分布圖


# In[1]:


# 載入基礎套件
import numpy as np
np.random.seed(1)

import matplotlib
import matplotlib.pyplot as plt



# ### 弱相關

# In[2]:


# 隨機生成兩組 1000 個介於 0~50 的數的整數 x, y, 看看相關矩陣如何
x = np.random.randint(0, 50, 1000)
y = np.random.randint(0, 50, 1000)

# 呼叫 numpy 裡的相關矩陣函數 (corrcoef)
print(np.corrcoef(x, y))


# In[3]:


# 將分布畫出來看看吧
plt.scatter(x, y)
plt.show()

# ### 正相關

# In[4]:


# 隨機生成 1000 個介於 0~50 的數 x
x = np.random.randint(0, 50, 1000)

# 這次讓 y 與 x 正相關，再增加一些雜訊
y = x + np.random.normal(0, 10, 1000)

# 再次用 numpy 裡的函數來計算相關係數
np.corrcoef(x, y)


# In[5]:


# 再看看正相關的 x,y 分布
plt.scatter(x, y)
plt.show()


# In[ ]:
x = np.random.randint(0 , 50 , 1000)
x = sorted(x)
y = x + np.linspace(50 , -50 , 1000)

print(np.corrcoef(x , y))
plt.scatter(x , y)
plt.show()


#!/usr/bin/env python
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測
# https://www.kaggle.com/c/titanic

# # [作業目標]
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 觀察填補缺值以及 標準化 / 最小最大化 對數值的影響

# # [作業重點]
# - 觀察替換不同補缺方式, 對於特徵的影響 (In[4]~In[6], Out[4]~Out[6])
# - 觀察替換不同特徵縮放方式, 對於特徵的影響 (In[7]~In[8], Out[7]~Out[8])

# In[1]:


# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression

data_path = 'D:/VScode workshop/3rd-ML100Days/practice/Day12/data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
print(df.head())


# In[2]:


#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')


# In[3]:


# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
train_num = train_Y.shape[0]
df.head()


# # 作業1
# * 試著在補空值區塊, 替換並執行兩種以上填補的缺值, 看看何者比較好?

# In[ ]:




# 空值補 -1, 做羅吉斯迴歸
df_m1 = df.fillna(-1)
train_X = df_m1[:train_num]
estimator = LogisticRegression()
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())

"""
Your Code Here
"""
df_mean = df.fillna(df.mean())
train_X = df_mean[:train_num]
print(cross_val_score(estimator , train_X , train_Y , cv = 5).mean())


df_mode = df
for i in num_features : 
	df_mode[i].fillna(df_mode[i].mode()[0] , inplace = True)

train_X = df_mode[:train_num]
print(cross_val_score(estimator , train_X , train_Y , cv = 5).mean())


# # 作業2
# * 使用不同的標準化方式 ( 原值 / 最小最大化 / 標準化 )，搭配羅吉斯迴歸模型，何者效果最好?

# In[ ]:


"""
Your Code Here
"""

#原值
estimator = LogisticRegression()
print(cross_val_score(estimator , train_X , train_Y , cv = 5).mean())

#min - max
for i in num_features : 
	val = df[i].values
	df[i] = (val  - min(val)) / max(val) - min(val)
print(cross_val_score(estimator , train_X , train_Y , cv = 5).mean())

#標準化
for i in num_features : 
	val = df[i].values
	df[i] = (val - val.mean()) / val.std()
print(cross_val_score(estimator , train_X , train_Y , cv = 5).mean())
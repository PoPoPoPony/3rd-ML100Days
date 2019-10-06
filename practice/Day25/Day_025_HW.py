#!/usr/bin/env python
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測
# https://www.kaggle.com/c/titanic

# # [作業目標]
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 觀察均值編碼的效果

# # [作業重點]
# - 仿造範例, 完成標籤編碼與均值編碼搭配邏輯斯迴歸的預測
# - 觀察標籤編碼與均值編碼在特徵數量 / 邏輯斯迴歸分數 / 邏輯斯迴歸時間上, 分別有什麼影響 (In[3], Out[3], In[4], Out[4]) 

# # 作業1
# * 請仿照範例，將鐵達尼範例中的類別型特徵改用均值編碼實作一次

# In[1]:


# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os
from copy import deepcopy


data_path = os.getcwd() + "/ml100_data/data/"
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')

train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
df.head()

# In[2]:


#只取類別值 (object) 型欄位, 存於 object_features 中
object_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'object':
        object_features.append(feature)
print(f'{len(object_features)} Numeric Features : {object_features}\n')

# 只留類別型欄位
df = df[object_features]
df = df.fillna('None')
train_num = train_Y.shape[0]
df.head()


# # 作業2
# * 觀察鐵達尼生存預測中，均值編碼與標籤編碼兩者比較，哪一個效果比較好? 可能的原因是什麼?

# In[ ]:



# 對照組 : 標籤編碼 + 邏輯斯迴歸

df_temp = pd.DataFrame()
LE = LabelEncoder()
for c in df.columns:
    df_temp[c] = LE.fit_transform(df[c])
train_X = df_temp[:train_num]
LR = LogisticRegression()
score_LB_LR = cross_val_score(LR , train_X , train_Y , cv = 5).mean()
print(score_LB_LR)




# In[ ]:


# 均值編碼 + 邏輯斯迴歸


df_temp = pd.concat([df[:train_num] , train_Y] , axis = 1)

for c in df.columns:
    mean_df = df_temp.groupby([c])['Survived'].mean().reset_index()
    mean_df.columns = [c, f'{c}_mean']
    df_temp = pd.merge(df_temp , mean_df, on=c, how='left')
    df_temp = df_temp.drop([c] , axis=1)
df_temp = df_temp.drop(['Survived'] , axis=1)

LR = LogisticRegression()
score_ME_LR = cross_val_score(LR , df_temp , train_Y , cv = 5).mean()
print(score_ME_LR)


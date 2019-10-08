#!/usr/bin/env python
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測

# # [作業目標]
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 練習特徵重要性的寫作與觀察

# # [作業重點]
# - 仿造範例, 完成特徵重要性的計算, 並觀察對預測結果的影響 (In[3]~[5], Out[3]~[5]) 
# - 仿造範例, 將兩個特徵重要性最高的特徵重組出新特徵, 並觀察對預測結果的影響 (In[8], Out[8]) 

# In[1]:


# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
import os


warnings.filterwarnings('ignore')

data_path = os.getcwd() + "/ml100_data/data/"
df = pd.read_csv(data_path + 'titanic_train.csv')

train_Y = df['Survived']
df = df.drop(['PassengerId', 'Survived'] , axis=1)
df.head()


# In[2]:


# 因為需要把類別型與數值型特徵都加入, 故使用最簡版的特徵工程
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
df.head()


# In[3]:


# 隨機森林擬合後, 將結果依照重要性由高到低排序
estimator = RandomForestClassifier()
estimator.fit(df.values, train_Y)
feats = pd.Series(data=estimator.feature_importances_, index=df.columns)
feats = feats.sort_values(ascending=False)
print(feats)


# ## 先用隨機森林對鐵達尼生存預測做訓練，再用其特徵重要性回答下列問題
# 
# # 作業1
# * 將特徵重要性較低的一半特徵刪除後，再做生存率預估，正確率是否有變化?

# In[ ]:


# 高重要性特徵 + 隨機森林
"""
Your Code Here
"""

high_feature = feats[:(feats.shape[0] // 2)].index
print(high_feature)

train_X = MMEncoder.fit_transform(df[high_feature])
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())


# In[ ]:


# 原始特徵 + 隨機森林

train_X = MMEncoder.fit_transform(df)
cross_val_score(estimator, train_X, train_Y, cv=5).mean()


# # 作業2
# * 將特徵重要性最高的兩個特徵做特徵組合，是否能再進一步提升預測力?

# In[ ]:


# 觀察重要特徵與目標的分布
# 第一名              


import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x=train_Y, y=df['Sex'], fit_reg=False)
plt.show()


# In[ ]:


# 第二名       

sns.regplot(x=train_Y, y=df['Ticket'], fit_reg=False)
plt.show()


# In[ ]:


# 製作新特徵看效果
"""
Your Code Here
"""
df['new_feature'] = (df['Sex'] + df['Ticket']) / 2

train_X = MMEncoder.fit_transform(df)
print(cross_val_score(estimator, train_X, train_Y, cv=5).mean())


# In[ ]:





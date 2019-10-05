#!/usr/bin/env python
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測精簡版 
# https://www.kaggle.com/c/titanic

# # [作業目標]
# - 試著不依賴說明, 只依照下列程式碼回答下列問題, 初步理解什麼是"特徵工程"的區塊

# # [作業重點]
# - 試著不依賴註解, 以之前所學, 回答下列問題

# # 作業1
# * 下列A~E五個程式區塊中，哪一塊是特徵工程?
# 
# # 作業2
# * 對照程式區塊 B 與 C 的結果，請問那些欄位屬於"類別型欄位"? (回答欄位英文名稱即可) 
# 
# # 作業3
# * 續上題，請問哪個欄位是"目標值"?

# In[ ]:


# 程式區塊 A
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import os

data_path = os.getcwd() + '/ml100_data/data/'
df_train = pd.read_csv(data_path + 'titanic_train.csv')
df_test = pd.read_csv(data_path + 'titanic_test.csv')
print(df_train.shape)


# In[ ]:


# 程式區塊 B
train_Y = df_train['Survived']
ids = df_test['PassengerId']
df_train = df_train.drop(['PassengerId', 'Survived'] , axis=1)
df_test = df_test.drop(['PassengerId'] , axis=1)
df = pd.concat([df_train,df_test])
print(df.head())


# In[ ]:


# 程式區塊 C
LEncoder = LabelEncoder()
MMEncoder = MinMaxScaler()
for c in df.columns:
    df[c] = df[c].fillna(-1)
    if df[c].dtype == 'object':
        df[c] = LEncoder.fit_transform(list(df[c].values))
    df[c] = MMEncoder.fit_transform(df[c].values.reshape(-1, 1))
print(df.head())


# In[ ]:


# 程式區塊 D
train_num = train_Y.shape[0]
train_X = df[:train_num]
test_X = df[train_num:]

from sklearn.linear_model import LogisticRegression
estimator = LogisticRegression()
estimator.fit(train_X, train_Y)
pred = estimator.predict(test_X)


# In[ ]:


# 程式區塊 E
sub = pd.DataFrame({'PassengerId': ids, 'Survived': pred})
sub.to_csv('titanic_baseline.csv', index=False) 


# In[ ]:

#1 : C
#2 : name、Sex、Ticket、cabin、Embarked
#3 : Survived


#!/usr/bin/env python
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測
# https://www.kaggle.com/c/titanic

# # [作業目標]
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 觀察標籤編碼與獨編碼熱的影響

# # [作業重點]
# - 回答在範例中的觀察結果
# - 觀察標籤編碼與獨熱編碼, 在特徵數量 / 邏輯斯迴歸分數 / 邏輯斯迴歸時間上, 分別有什麼影響 (In[3], Out[3], In[4], Out[4]) 

# # 作業1
# * 觀察範例，在房價預測中調整標籤編碼(Label Encoder) / 獨熱編碼 (One Hot Encoder) 方式，  
# 對於線性迴歸以及梯度提升樹兩種模型，何者影響比較大?

# In[1]:

#1 : 應該是回歸吧ˊˇˋ

# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy, time
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import os



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
df2 = copy.deepcopy(df)

# # 作業2
# * 鐵達尼號例題中，標籤編碼 / 獨熱編碼又分別對預測結果有何影響? (Hint : 參考今日範例)

# In[ ]:


# 標籤編碼 + 羅吉斯迴歸
LE = LabelEncoder()
for i in df.columns : 
	df[i] = LE.fit_transform(df[i])

train_X = df[:train_num]
est = LogisticRegression()
score_lb_lg = cross_val_score(est , train_X , train_Y , cv = 5).mean()
print(score_lb_lg)

"""
Your Code Here
"""


# In[ ]:


# 獨熱編碼 + 羅吉斯迴歸
df2 = pd.get_dummies(df2)
train_X = df[:train_num]
est2 = LogisticRegression()
score_oh_lg = cross_val_score(est2 , train_X , train_Y , cv = 5).mean()
print(score_oh_lg)


"""
Your Code Here
"""


# In[ ]:





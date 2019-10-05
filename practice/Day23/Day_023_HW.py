#!/usr/bin/env python
# coding: utf-8

# # 作業 : (Kaggle)鐵達尼生存預測 
# https://www.kaggle.com/c/titanic

# # [作業目標]
# - 試著模仿範例寫法, 在鐵達尼生存預測中, 觀察降低偏態的影響

# # [作業重點]
# - 觀察使用log1p降偏態時, 對於分布與迴歸分數的影響 (In[6], Out[6])
# - 修正區塊中的資料問題後, 觀察以box-cox降偏態, 對於分布與迴歸分數的影響 (In[7], Out[7])

# In[1]:


# 做完特徵工程前的所有準備 (與前範例相同)
import pandas as pd
import numpy as np
import copy
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
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


#只取 int64, float64 兩種數值型欄位, 存於 num_features 中
num_features = []
for dtype, feature in zip(df.dtypes, df.columns):
    if dtype == 'float64' or dtype == 'int64':
        num_features.append(feature)
print(f'{len(num_features)} Numeric Features : {num_features}\n')

# 削減文字型欄位, 只剩數值型欄位
df = df[num_features]
df = df.fillna(0)
MMEncoder = MinMaxScaler()
train_num = train_Y.shape[0]
df.head()


# In[3]:


# 顯示 Fare 與目標值的散佈圖
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(df['Fare'][:train_num])
plt.show()


# In[4]:


# 計算基礎分數
df_mm = MMEncoder.fit_transform(df)
train_X = df_mm[:train_num]
estimator = LogisticRegression()
score = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print(score)

# # 作業1 
# * 試著在鐵達尼的票價 (Fare) 欄位中使用對數去偏 (log1p) , 結果是否更好?

# In[ ]:


# 將 Fare 取 log1p 後, 看散佈圖, 並計算分數
df_fixed = copy.deepcopy(df)

df_fixed['Fare'] = np.log1p(df_fixed['Fare'])

sns.distplot(df_fixed['Fare'][:train_num])
plt.show()

df_fixed = MMEncoder.fit_transform(df_fixed)
train_X = df_fixed[:train_num]
estimator = LogisticRegression()
score_fix = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print(score_fix)

#結果有稍微變好


# # 作業2
# * 最後的 boxcox 區塊直接執行會造成錯誤, 起因為輸入值有負值, 請問如何修正後可以使用 boxcox? (Hint : 試圖修正資料)

# In[ ]:



# 將 Fare 取 boxcox 後, 看散佈圖, 並計算分數 (執行會有 error, 請試圖修正)
from scipy import stats
from sklearn.preprocessing import Imputer

df_fixed = copy.deepcopy(df)
"""
Your Code Here, fix the error
"""

print(df_fixed['Fare'].describe())
imputer = Imputer(missing_values = 0 , strategy = 'most_frequent' , copy = True)
df_fixed[['Fare']] = imputer.fit_transform(df_fixed[['Fare']])


df_fixed['Fare'] = stats.boxcox(df_fixed['Fare'])[0]
sns.distplot(df_fixed['Fare'][:train_num])
plt.show()

df_fixed = MMEncoder.fit_transform(df_fixed)
train_X = df_fixed[:train_num]
estimator = LogisticRegression()
score_fix2 = cross_val_score(estimator, train_X, train_Y, cv=5).mean()
print(score_fix2)

# In[ ]:





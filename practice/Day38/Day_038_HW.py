#!/usr/bin/env python
# coding: utf-8

# ## [作業重點]
# 使用 Sklearn 中的線性迴歸模型，來訓練各種資料集，務必了解送進去模型訓練的**資料型態**為何，也請了解模型中各項參數的意義

# ## 作業
# 試著使用 sklearn datasets 的其他資料集 (wine, boston, ...)，來訓練自己的線性迴歸模型。

# ### HINT: 注意 label 的型態，確定資料集的目標是分類還是回歸，在使用正確的模型訓練！

# In[ ]:

from sklearn import datasets
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


wine = datasets.load_wine()
df = pd.DataFrame(wine['data'] , columns = wine['feature_names'])
temp = df['proline']
df = df.drop('proline' , axis = 1)

train_X , test_X , train_Y , test_Y = train_test_split(df , temp , test_size = 0.1)

LR = LinearRegression()
LR.fit(train_X , train_Y)
pred_Y = LR.predict(test_X)
mse = mean_squared_error(test_Y , pred_Y)
mae = mean_absolute_error(test_Y , pred_Y)
r2 = r2_score(test_Y , pred_Y)

print("mse : {}".format(mse))
print("mae : {}".format(mae))
print("r2 : {}".format(r2))
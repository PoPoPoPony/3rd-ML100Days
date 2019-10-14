#!/usr/bin/env python
# coding: utf-8

# ## [作業重點]
# 使用 Sklearn 中的 Lasso, Ridge 模型，來訓練各種資料集，務必了解送進去模型訓練的**資料型態**為何，也請了解模型中各項參數的意義。
# 
# 機器學習的模型非常多種，但要訓練的資料多半有固定的格式，確保你了解訓練資料的格式為何，這樣在應用新模型時，就能夠最快的上手開始訓練！

# ## 練習時間
# 試著使用 sklearn datasets 的其他資料集 (boston, ...)，來訓練自己的線性迴歸模型，並加上適當的正則化來觀察訓練情形。

# In[ ]:

import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso , Ridge
from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error


wine = datasets.load_wine()
df = pd.DataFrame(wine['data'] , columns = wine['feature_names'])
temp = df['proline']
df = df.drop('proline' , axis = 1)

train_X , test_X , train_Y , test_Y = train_test_split(df , temp , test_size = 0.1)
LR = Lasso(alpha = 3)
LR.fit(train_X , train_Y)
pred_Y = LR.predict(test_X)

r2 = r2_score(test_Y , pred_Y)
mae = mean_absolute_error(test_Y , pred_Y)
mse = mean_squared_error(test_Y , pred_Y)

print("L1 r2 : {}".format(r2))
print("L1 mae : {}".format(mae))
print("L1 mse : {}".format(mse))


LR = Ridge(alpha = 3)
LR.fit(train_X , train_Y)
pred_Y = LR.predict(test_X)

r2 = r2_score(test_Y , pred_Y)
mae = mean_absolute_error(test_Y , pred_Y)
mse = mean_squared_error(test_Y , pred_Y)

print("L2 r2 : {}".format(r2))
print("L2 mae : {}".format(mae))
print("L2 mse : {}".format(mse))







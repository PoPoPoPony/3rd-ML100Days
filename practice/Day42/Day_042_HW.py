#!/usr/bin/env python
# coding: utf-8

# ## [作業重點]
# 目前你應該已經要很清楚資料集中，資料的型態是什麼樣子囉！包含特徵 (features) 與標籤 (labels)。因此要記得未來不管什麼專案，必須要把資料清理成相同的格式，才能送進模型訓練。
# 今天的作業開始踏入決策樹這個非常重要的模型，請務必確保你理解模型中每個超參數的意思，並試著調整看看，對最終預測結果的影響為何

# ## 作業
# 
# 1. 試著調整 DecisionTreeClassifier(...) 中的參數，並觀察是否會改變結果？
# 2. 改用其他資料集 (boston, wine)，並與回歸模型的結果進行比較

# In[ ]:

import pandas as pd
from sklearn import datasets
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

wine = datasets.load_wine()
df = pd.DataFrame(wine['data'] , columns = wine['feature_names'])
temp = df['proline']
df = df.drop('proline' , axis = 1)

train_X , test_X , train_Y , test_Y = train_test_split(df , temp , test_size = 0.1)

#decision tree regressor
dtr = DecisionTreeRegressor(min_samples_leaf = 5 , criterion = 'mae')
dtr.fit(train_X , train_Y)
pred_Y = dtr.predict(test_X)

mae = mean_absolute_error(test_Y , pred_Y)
mse = mean_squared_error(test_Y , pred_Y)
r2 = r2_score(test_Y , pred_Y)

print("dtr mae : {}".format(mae))
print("dtr mse : {}".format(mse))
print("dtr r2 : {}".format(r2))

print(dtr.feature_importances_)


#ridge regression
ridge = Ridge(alpha = 3)
ridge.fit(train_X , train_Y)
pred_Y = ridge.predict(test_X)

mae = mean_absolute_error(test_Y , pred_Y)
mse = mean_squared_error(test_Y , pred_Y)
r2 = r2_score(test_Y , pred_Y)

print("ridge mae : {}".format(mae))
print("ridge mse : {}".format(mse))
print("ridge r2 : {}".format(r2))




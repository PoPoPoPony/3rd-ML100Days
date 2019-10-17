#!/usr/bin/env python
# coding: utf-8

# ## [作業重點]
# 確保你了解隨機森林模型中每個超參數的意義，並觀察調整超參數對結果的影響

# ## 作業
# 
# 1. 試著調整 RandomForestClassifier(...) 中的參數，並觀察是否會改變結果？
# 2. 改用其他資料集 (boston, wine)，並與回歸模型與決策樹的結果進行比較

# In[ ]:
import pandas as pd
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor
from sklearn.metrics import mean_squared_error , mean_absolute_error , r2_score , classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


wine = datasets.load_wine()
df = pd.DataFrame(wine['data'] , columns = wine['feature_names'])
temp = df['proline']
df = df.drop('proline' , axis = 1)

train_X , test_X , train_Y , test_Y = train_test_split(df , temp , test_size = 0.1)

#Random forest regressor
clf = RandomForestRegressor(min_samples_leaf = 5 , criterion = 'mae')
clf.fit(train_X , train_Y)
pred_Y = clf.predict(test_X)

mae = mean_absolute_error(test_Y , pred_Y)
mse = mean_squared_error(test_Y , pred_Y)
r2 = r2_score(test_Y , pred_Y)

print("clf mae : {}".format(mae))
print("clf mse : {}".format(mse))
print("clf r2 : {}".format(r2))


print(clf.feature_importances_)



bc = datasets.load_breast_cancer()

df_X = pd.DataFrame(bc['data'] , columns = bc['feature_names'])
df_Y = pd.DataFrame(bc['target'] , columns = ['malignant'])

train_X , test_X , train_Y , test_Y = train_test_split(df_X , df_Y , test_size = 0.1)

clf = RandomForestClassifier(criterion = 'entropy' , min_samples_leaf = 5)
clf.fit(train_X , train_Y)
pred_Y = clf.predict(test_X)

rp = classification_report(test_Y , pred_Y)
print(rp)

#!/usr/bin/env python
# coding: utf-8

# ## [作業重點]
# 了解如何使用 Sklearn 中的 hyper-parameter search 找出最佳的超參數

# ### 作業
# 請使用不同的資料集，並使用 hyper-parameter search 的方式，看能不能找出最佳的超參數組合

# In[ ]:


import pandas as pd
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

digits = datasets.load_digits()


df_X = pd.DataFrame(digits['data'])
df_Y = pd.DataFrame(digits['target'])

print(df_X.shape)


train_X , test_X , train_Y , test_Y = train_test_split(df_X , df_Y , test_size = 0.2)
clf = GradientBoostingClassifier()
lr = [0.1 , 0.2 , 0.3]
n_estimator = [150 , 100 , 50]

param_grid = dict(learning_rate = lr , n_estimators = n_estimator)

grid_search = GridSearchCV(clf , param_grid , scoring = 'accuracy')
grid_result = grid_search.fit(train_X , train_Y)
print(grid_result.best_score_)
print(grid_result.best_params_)
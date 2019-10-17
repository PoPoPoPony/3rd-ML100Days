#!/usr/bin/env python
# coding: utf-8

# ### 作業
# 目前已經學過許多的模型，相信大家對整體流程應該比較掌握了，這次作業請改用**手寫辨識資料集**，步驟流程都是一樣的，請試著自己撰寫程式碼來完成所有步驟

# In[2]:

import pandas as pd
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report



digits = datasets.load_digits()


# In[ ]:


df_X = pd.DataFrame(digits['data'])
df_Y = pd.DataFrame(digits['target'])

print(df_X.shape)


train_X , test_X , train_Y , test_Y = train_test_split(df_X , df_Y , test_size = 0.2)
clf = GradientBoostingClassifier(learning_rate = 0.1 , n_estimators =  100)

clf.fit(train_X , train_Y)
pred_Y = clf.predict(test_X)

score = classification_report(test_Y , pred_Y)
print(score)

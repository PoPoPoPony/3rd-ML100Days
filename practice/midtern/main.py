import preprocessing as pre
import model
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import os

def write_file(pred_y , data_path) : 
    output = pd.DataFrame({'name' : name_df['name'] , 'poi' : pred_y})
    output.to_csv(data_path + "/pred.csv" , index = False)

# , index = range(pred_y.shape[0]))

data_path = os.getcwd() + "/midtern/data"
train_df , test_df = pre.read_file(data_path)

name_df = test_df[['name']]
train_df['poi'] = train_df['poi'].map(lambda x : 1 if x is True else 0)
corr_col = pre.corr_test(train_df)


train_y = train_df[['poi']]
train_df = train_df.drop('poi' , axis = 1)
corr_col.remove('poi')

train_num = train_df.shape[0]
train_df = train_df[corr_col]
test_df = test_df[corr_col]

df = pd.concat([train_df , test_df])
df = df.reset_index()
df = df.drop('index' , axis = 1)

#df = df.drop(['name' , 'email_address'] , axis = 1)


#df = pre.drop_data(df)


df = pre.fill_na(df)

df = pre.mm_scaler(df)

train_X = df[ : train_num]
test_X = df[train_num : ]

#model.cv_score(train_X , train_y)

#lr = LogisticRegression(solver = 'lbfgs')
#rf = RandomForestClassifier(max_depth = 10 , min_samples_leaf = 5)

#model.model_test(train_X , train_y , lr , rf)

'''
pred_y = model.lr(train_X , train_y , test_X)
print(pred_y)
write_file(pred_y , data_path)
'''


pred_y = model.blending(train_X , train_y , test_X , 0.35 , 0.65)
print(pred_y)
write_file(pred_y , data_path)


'''
pred_y = model.stacking(train_X , train_y , test_X)
print(pred_y)
write_file(pred_y , data_path)
'''

'''
param , score = model.search('rf' , train_X , train_y)
print("rf {} , score : {}".format(param , score))
param , score = model.search('gdbt' , train_X , train_y)
print("gdbt {} , score : {}".format(param , score))
'''



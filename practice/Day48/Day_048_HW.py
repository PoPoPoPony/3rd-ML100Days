import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


data_path = os.getcwd() + "/practice/Day48/data/"
col = []

for i in range(1 , 41) : 
    col.append("C" + str(i))


try : 
    train_df = pd.read_csv(data_path + "train.csv" , header = None)
    train_df.columns = col
    train_df['target'] = pd.read_csv(data_path + "trainLabels.csv" , header = None)
    test_df = pd.read_csv(data_path + "test.csv" , header = None)
    test_df.columns = col
except : 
    print("read file error")


train_Y = train_df['target']
train_df = train_df.drop('target' , axis = 1)

train_num = train_df.shape[0]

df_X = pd.concat([train_df , test_df])

mmScaler = MinMaxScaler()
df = mmScaler.fit_transform(df_X)

train_X = df[:train_num]
test_X = df[train_num:]

lr = LogisticRegression()
lr.fit(train_X , train_Y)
pred_Y = lr.predict(test_X)

output = pd.DataFrame(pred_Y , columns = ['Solution'])
output.index = list(range(1 , 9001))
print(output.tail())
output.to_csv(data_path + "predition.csv")
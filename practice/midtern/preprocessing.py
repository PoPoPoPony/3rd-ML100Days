import pandas as pd
import os
import numpy as np
from sklearn.preprocessing import Imputer , MinMaxScaler
from matplotlib import pyplot as plt



def read_file(data_path) : 
    try : 
        train_df = pd.read_csv(data_path + "/train_data.csv")
        test_df = pd.read_csv(data_path + "/test_features.csv")
        #y_df = pd.read_csv(data_path + "/sample_submission.csv")
        return train_df , test_df
    except : 
        print("read file error")

def null_det(df) : 
    store = dict()
    for i in df.columns : 
        null_count = df[i].isnull().sum()
        if null_count > 0 : 
            store[i] = null_count
    
    
    for i , j in store.items() : 
        print("{} has {} null values".format(i , j))
    
    return store

#先丟掉缺失值數量 >= 80 的col
def drop_data(df) : 
    null_data = null_det(df)
    for i , j in null_data.items() : 
        if j >= 100 : 
            df = df.drop(i , axis = 1)
    return df


def fill_na(df) : 
    temp_df = df[['loan_advances' , 'deferred_income' , 'deferral_payments']]
    temp_df['loan_advances'] = temp_df['loan_advances'].fillna(0)
    temp_df['loan_advances'] = temp_df['loan_advances'].map(lambda x : 1 if x != 0.0 else 0)
    temp_df['deferred_income'] = temp_df['deferred_income'].fillna(0)
    temp_df['deferred_income'] = temp_df['deferred_income'].map(lambda x : 1 if x != 0.0 else 0)
    temp_df['deferral_payments'] = temp_df['deferral_payments'].fillna(0)
    temp_df['deferral_payments'] = temp_df['deferral_payments'].map(lambda x : 1 if x != 0.0 else 0)
    
    imp = Imputer(missing_values = np.nan , strategy = "most_frequent" , copy = False)
    imp.fit_transform(df)

    df['loan_advances'] = temp_df['loan_advances']
    df['deferred_income'] = temp_df['deferred_income']
    df['deferral_payments'] = temp_df['deferral_payments']
    return df

def mm_scaler(df) : 
    mm = MinMaxScaler(copy = False)
    mm.fit_transform(df)
    return df

def corr_test(df) : 
    cor_df = df.corr()
    x = cor_df['poi'].abs().sort_values(ascending = False).index.to_list()[:15]
    return x


def draw_scatter(df , col) : 
    plt.scatter(range(df.shape[0]) , df[col])
    plt.xlabel("data order")
    plt.ylabel(col)
    plt.title("EDA scatter")
    plt.show()
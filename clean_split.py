import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
import random 
from pprint import pprint

def clean_data(df):
    df = df.drop("ID",axis=1)
    df.dropna(inplace=True)
    df.drop_duplicates(inplace = True)
    df = df.rename(columns={"Segmentation": "label"})
    return df

def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df    

def train_valid_test_split(df,test_size):
    train_df,test_df=train_test_split(df,test_size)
    train_df,valid_df=train_test_split(train_df,0.167)
    return train_df,valid_df,test_df    
import pandas as pd
from sklearn.model_selection import train_test_split
from kfp.dsl import component, OutputPath, Dataset
from typing import NamedTuple

def prepare_data(*,train: OutputPath(str), test: OutputPath(str)):
    # Load dataset 
    df = pd.read_csv('data/transactions.csv')

    # # separate dataset into train and test
    # from sklearn.model_selection import train_test_split
    train_df,test_df = train_test_split(df,test_size=0.2,random_state=42)

    # Save to files
    train = 'train.csv'
    test = 'test.csv'
    train_df.to_csv('train.csv',index=False)
    test_df.to_csv('test.csv',index=False)

    return 'train.csv', 'test.csv'
    
    


if __name__ == '__main__':
    prepare_data()
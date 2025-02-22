import pandas as pd
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score,precision_score,recall_score
from kfp.dsl import component, OutputPath, Dataset

import json


def evaluate_model(*,model_path,train_data_path,test_data_path,metrics:OutputPath('JSON')):
    
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Load training data
    train_data = pd.read_csv(train_data_path)
    X_train= train_data.drop('fraud',axis=1)
    y_train = train_data['fraud']

    # Load test data
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop('fraud',axis=1)
    y_test = test_data['fraud']
    

    # # Create Column Transformer with 3 types of transformers
    num_features = X_train.select_dtypes(exclude="object").columns
    cat_features = ['gender',  'type', 'Card Type', 'Exp Type']


    numeric_transformer = StandardScaler()
    oh_transformer = OneHotEncoder()

    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", oh_transformer, cat_features),
            ("StandardScaler", numeric_transformer, num_features),        
        ]
    )

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)

    ## Make predictions

    y_pred = model.predict(X_test)

    # Calculate metrics

    metrics = {
        'f1_score' : f1_score(y_test, y_pred),
        'recall_score' : recall_score(y_test, y_pred),
        'precision_score': precision_score(y_test,y_pred)
    }

    ## Save metrics

    with open('metrics.json','w') as f:
        json.dump(metrics,f)
    
    return 'metrics.json'


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str)
    parser.add_argument('--train-data',type=str)
    parser.add_argument('--test-data',type=str)
    args = parser.parse_args()
    evaluate_model(args.model,args.train_data,args.test_data)

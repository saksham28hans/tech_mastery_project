import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier
)
from sklearn.model_selection import GridSearchCV
from kfp.dsl import component, OutputPath, Dataset, Output

def evaluate_models(X_train,y_train,X_test,y_test,models,param):
    report = {}
    print(X_train[0:5])
    print(X_test[0:5])
    for i in range(len(list(models))):
        model = list(models.values())[i]
        para=param[list(models.keys())[i]]

        gs = GridSearchCV(model,para,cv=3)
        gs.fit(X_train,y_train)
        
        model.set_params(**gs.best_params_)
        model.fit(X_train,y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_model_score = r2_score(y_train,y_train_pred)
        test_model_score  = r2_score(y_test,y_test_pred)

        report[list(models.keys())[i]] = test_model_score
        
    print(report)
    return report
    

def train_model(*,train_data_path,test_data_path,model:OutputPath(str)):
# def train_model(train_data_path,test_data_path):
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

    models = {
                "Random Forest" : RandomForestClassifier(verbose=1),
                "Decision Tree" : DecisionTreeClassifier(),
                "Gradient Boosting" : GradientBoostingClassifier(verbose=1),
                "Logistic Regression": LogisticRegression(verbose=1),
                "AdaBoost Classifier" : AdaBoostClassifier()
            }
    
    params={
            "Random Forest":{
                'n_estimators': [8,16,32,128,256]
            },
            "Decision Tree": {
                'criterion':['gini', 'entropy', 'log_loss'],
            },
            "Gradient Boosting":{
                # 'learning_rate':[.1,.01,.05],
                # 'n_estimators': [50,100,200],
                # 'max_depth' : [3,5,7]
            },
            "Logistic Regression":{},
            "AdaBoost Classifier":{
            }
            
        }

    model_report : dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models,param=params)

    ## To get the best model score from dict
    best_model_score = max((sorted(model_report.values())))

    ## To get the best model name  from dict

    best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
    best_model = models[best_model_name]

    ## Save model

    with open('model.pkl','wb') as f:
        pickle.dump(best_model,f)
    
    return 'model.pkl'

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data',type=str)
    parser.add_argument('--test-data',type=str)
    args = parser.parse_args()
    train_model(args.train_data,args.test_data)



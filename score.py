import json
import numpy
import pandas as pd
from azureml.core.model import Model
import pickle

def init():
    global model
    model_path = Model.get_model_path('fraud-predictor')

    with open(model_path,'rb') as f:
        model = pickle.load(f)


def run(raw_data):
    try:
        data = json.loads(raw_data)
        data = pd.DataFrame(data)
        prediction = model.predict(data)
        return json.dumps({'prediction':prediction.tolist()})
    except Exception as e:
        return json.dumps({'error':str(e)})
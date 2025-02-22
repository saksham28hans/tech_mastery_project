from kfp import dsl
from kfp import components
from kfp.dsl import component, ContainerSpec, OutputPath, container_component,Output



@component
def prepare_data_op(*,train: OutputPath(str), test: OutputPath(str)):
    return dsl.ContainerSpec(
        image = 'python:3.8',
        command = ['python', 'dataset_prep.py'],
        args =[train,test]
        
    )

@component
def train_model_op(*,train_data:str,test_data:str,model:OutputPath(str)):
    return dsl.ContainerSpec(
        image = 'python:3.8',
        command = ['python', 'train.py'],
        args = ['--train-data',train_data,
                     '--test-data',test_data,
                     model
                     ],
    )

@component
def evaluate_model_op(*,model:str,train_data:str,test_data:str,metrics:OutputPath('JSON')) -> str:
    return dsl.ContainerSpec(
        image = 'python:3.8',
        command = ['python', 'evaluate.py'],
        args = ['--model',model,
                     '--train-data',train_data,
                     '--test-data',test_data,
                     metrics
                     ],
    )

@dsl.pipeline(
    name = 'Fraud Detection Pipeline',
    description='Train and evaluate a bank fraud detection model'
)

def fraud_detection_pipeline():
    data_prep = prepare_data_op()
    train = train_model_op(train_data=data_prep.outputs['train'],test_data=data_prep.outputs['test'])
    evaluate = evaluate_model_op(
        model=train.outputs['model'],
        train_data=data_prep.outputs['train'],
        test_data=data_prep.outputs['test']
    )

# Compile Pipeline
from kfp.compiler import Compiler
Compiler().compile(fraud_detection_pipeline,'fraud_detection_pipeline.yaml')
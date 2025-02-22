from azureml.core import Workspace,Model,Environment
from azureml.core.webservice import AciWebservice
from azureml.core.model import InferenceConfig

def deploy_model(model_path):

    ws = Workspace.from_config()

    model = Model.register(
        workspace = ws,
        model_path = model_path,
        model_name = 'fraud-predictor'
    )

    env = Environment.from_conda_specification(
        name = 'fraud-env',
        file_path = 'environment.yml'
    )

    inference_config = InferenceConfig(
        entry_script = 'score.py',
        environment = env
    )

    deployment_config = AciWebservice.deploy_configuration(
        cpu_cores = 1,
        memory_gb = 1,
        auth_enable = True
    )

    service = Model.deploy(
        workspace = ws,
        name = 'fraud-service',
        models = [model],
        inference_config = inference_config,
        deployment_config = deployment_config
    )

    service.wait_for_deployment(show_output=True)
    return service.scoring_uri

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',type=str)
    args = parser.parse_args()
    deploy_model(args.model)
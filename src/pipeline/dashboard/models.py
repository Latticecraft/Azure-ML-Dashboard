#%%
import os, argparse
import json
import mlflow

from azureml.core import Model, Run
from distutils.dir_util import copy_tree


# define functions 
def main(ctx):
    models = Model.list(workspace=ctx['run'].experiment.workspace, latest=True)
    json_models = [{'name': m.name, 'type': models[i].tags['type']} for i,m in enumerate(models)]

    with open('outputs/models.json', 'w') as f:
        json.dump(json_models, f)
    
    copy_tree('outputs', args.transformed_data)


def start(args):
    os.makedirs("outputs", exist_ok=True)
    mlflow.start_run()
    client = mlflow.tracking.MlflowClient()
    run = Run.get_context()
    tags = run.parent.get_tags()
    return {
        'args': args,
        'run': run,
        'data': client.get_run(mlflow.active_run().info.run_id).data,
        'project': tags['project'],
        'type': tags['type'],
        'label': tags['label']
    }


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--datasets-pkl", type=str, default='data')
    parser.add_argument('--runinfo', type=str)
    parser.add_argument('--trainlog', type=str)
    parser.add_argument('--transformed-data', type=str)
    
    # parse args
    args = parser.parse_args()

    # return args
    return args


# run script
if __name__ == "__main__":
    # parse args
    args = parse_args()
    ctx = start(args)

    # run main function
    main(ctx)
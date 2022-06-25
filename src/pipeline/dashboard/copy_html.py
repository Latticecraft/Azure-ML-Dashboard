# imports
import os, argparse
import numpy as np
import pandas as pd
import mlflow

from azureml.core import Dataset, Datastore, Run
from azureml.data.datapath import DataPath
from datetime import datetime
from distutils.dir_util import copy_tree
from pathlib import Path


# define functions     
def main(ctx):
    # copy inputs to output
    datastore = Datastore.get(ctx['run'].experiment.workspace, 'output')

    Dataset.File.upload_directory(src_dir=ctx['args'].input1,
        target=DataPath(datastore, f'{ctx["project"]}/html/latest'),
        overwrite=True)

    Dataset.File.upload_directory(src_dir=ctx['args'].input2,
        target=DataPath(datastore, f'{ctx["project"]}/html/latest'),
        overwrite=True)

    Dataset.File.upload_directory(src_dir=ctx['args'].input3,
        target=DataPath(datastore, f'{ctx["project"]}/html/latest'),
        overwrite=True)

    Dataset.File.upload_directory(src_dir=ctx['args'].input4,
        target=DataPath(datastore, f'{ctx["project"]}/html/latest'),
        overwrite=True)

    Dataset.File.upload_directory(src_dir=ctx['args'].input5,
        target=DataPath(datastore, f'{ctx["project"]}/html/latest'),
        overwrite=True)

    now = datetime.now()
    ts = now.strftime('%m%d%y')

    Dataset.File.upload_directory(src_dir=ctx['args'].input1,
        target=DataPath(datastore, f'{ctx["project"]}/html/{ts}'),
        overwrite=True)

    Dataset.File.upload_directory(src_dir=ctx['args'].input2,
        target=DataPath(datastore, f'{ctx["project"]}/html/{ts}'),
        overwrite=True)

    Dataset.File.upload_directory(src_dir=ctx['args'].input3,
        target=DataPath(datastore, f'{ctx["project"]}/html/{ts}'),
        overwrite=True)

    Dataset.File.upload_directory(src_dir=ctx['args'].input4,
        target=DataPath(datastore, f'{ctx["project"]}/html/{ts}'),
        overwrite=True)

    Dataset.File.upload_directory(src_dir=ctx['args'].input5,
        target=DataPath(datastore, f'{ctx["project"]}/html/{ts}'),
        overwrite=True)
    

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
        'project': tags['project']
    }


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument("--input1", type=str)
    parser.add_argument("--input2", type=str)
    parser.add_argument("--input3", type=str)
    parser.add_argument("--input4", type=str)
    parser.add_argument("--input5", type=str)
    parser.add_argument("--html", type=str)

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
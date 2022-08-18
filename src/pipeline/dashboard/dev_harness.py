#%%
import os, joblib, glob, re
import numpy as np
import pandas as pd
import bokeh.palettes as bp
import holoviews as hv
import holoviews.operation.datashader as hd
import json

from azureml.core import Workspace, Datastore, Dataset, Run
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.data.datapath import DataPath
from datetime import datetime, timedelta
from interpret.ext.blackbox import TabularExplainer
from pathlib import Path
from holoviews.operation.datashader import spread, rasterize

hv.extension('bokeh')


def get_df(ds, path):
    try:
        df_all = pd.DataFrame()
        ds_path = Dataset.File.from_files(path=DataPath(ds, path))
        with ds_path.mount() as mount_context:
            print(f'loading df from {os.listdir(mount_context.mount_point)}')

            deltas = glob.glob(mount_context.mount_point+"/*")
            for d in deltas:
                print('adding {}'.format(d))
                df_delta = pd.read_csv((Path(mount_context.mount_point) / d), parse_dates=['runDate'])
                df_all = pd.concat([df_all, df_delta], ignore_index=True)

        df_all['runDate'] = pd.to_datetime(df_all['runDate'])
        return df_all.sort_values('runDate', ascending=False)
    except:
        return pd.DataFrame()


# connect to workspace

auth = InteractiveLoginAuthentication(tenant_id='')

ws = Workspace(subscription_id='',
               resource_group='',
               workspace_name='',
               auth=auth)

ds = Datastore.get(ws, 'output')
df_runinfo = get_df(ds, '<Project Folder>/runinfo')
df_trainlog = get_df(ds, '<Project Folder>/trainlog')

# theme
cmap_greens = bp.Greens[256][::-1][64:]

# variables
runId = ''
model, dict_files, explainer = None, {}, None
if len(df_trainlog) > 0:
    runId = df_trainlog.sort_values('runDate', ascending=False).iloc[0]['sweep_get_best_model_runid']
    
    # download latest run artifacts
    run = Run.get(workspace=ws, run_id=runId)
    run.download_file('outputs/model.pkl', output_file_path='data/model.pkl')
    run.download_file('outputs/datasets.pkl', output_file_path='data/datasets.pkl')
    run.download_file('outputs/best_run.json', output_file_path='data/best_run.json')

    with open('data/best_run.json', 'r') as f:
        best_run = json.load(f)
        imputer = best_run['imputer']
        balancer = best_run['balancer']

    model = joblib.load('data/model.pkl')
    dict_files = pd.read_pickle('data/datasets.pkl')
    data = LazyEval(dict_files)

    X_train, y_train = data.get('train', imputer, balancer)
    X_valid, y_valid = data.get('valid', imputer, balancer)
    X_test, y_test = data.get('test', imputer, balancer)

    explainer = TabularExplainer(model, X_train)

    dict_new = {
        'X_train': X_train,
        'y_train': y_train,
        'X_valid': X_valid,
        'y_valid': y_valid,
        'X_test': X_test,
        'y_test': y_test
    }

#%%
'''
    Use this section as a test harness to develop and test visualization code
    to be included in the HTML5 dashboard 
'''
from holoviews import dim, opts



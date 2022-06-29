#%%
import os, joblib, glob, re
import numpy as np
import pandas as pd
import bokeh.palettes as bp
import holoviews as hv
import holoviews.operation.datashader as hd

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
auth = InteractiveLoginAuthentication(tenant_id='9c8b18d3-0d09-4525-a546-341d38f12190')

ws = Workspace(subscription_id='83b4b5c6-51ae-4d5a-a7cf-63d20ffc2754',
               resource_group='MLproduct23',
               workspace_name='ltcml23',
               auth=auth)

ds = Datastore.get(ws, 'output')
df_runinfo = get_df(ds, 'Boston-House-Prices/runinfo')
df_trainlog = get_df(ds, 'Boston-House-Prices/trainlog')

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

    model = joblib.load('data/model.pkl')
    dict_files = pd.read_pickle('data/datasets.pkl')
    explainer = TabularExplainer(model, dict_files['X_train'])

#%%
'''
    Use this function as a test harness to develop and test visualization code
    to be included in the HTML5 dashboard 
'''
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error


def get_residuals_plot(ctx, model, dict_files):
    # get metrics
    yhat = model.predict(dict_files['X_test'])
    residuals = dict_files['y_test']['MEDV'] - yhat
    explained_variance = explained_variance_score(dict_files['y_test'], yhat)
    mae = mean_absolute_error(dict_files['y_test'], yhat)
    rmse = mean_squared_error(dict_files['y_test'], yhat, squared=False)
    r2 = r2_score(dict_files['y_test'], yhat)
    mape = mean_absolute_percentage_error(dict_files['y_test'], yhat)

    # visualize
    opts_dist = dict(width=450, height=450)
    opts_metrics = dict(width=450, height=450, xlim=(0,1), ylim=(0,1), xaxis='bare', yaxis='bare')

    dist = hv.Distribution(residuals).opts(**opts_dist)

    metrics = (hv.Text(0.3,0.9,'Explained Variance') *
        hv.Text(0.3,0.85,str(round(explained_variance, 3))) *
        hv.Text(0.3,0.7,'MAE') *
        hv.Text(0.3,0.65,str(round(mae, 3))) *
        hv.Text(0.3,0.5,'RMSE') *
        hv.Text(0.3,0.45,str(round(rmse, 3))) *
        hv.Text(0.3,0.3,'R^2') *
        hv.Text(0.3,0.25,str(round(r2, 3))) *
        hv.Text(0.3,0.1,'MAPE') *
        hv.Text(0.3,0.05,str(round(mape, 3)))).opts(**opts_metrics)
        
    overlay = dist + metrics

    return overlay

def get_viz():
    '''
        Begin visualization code
    '''
    viz = get_residuals_plot(None, model, dict_files)
    return viz
    

    '''
        End visualization code
    '''

viz = get_viz()
viz

# %%

# %%

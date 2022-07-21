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
'''
auth = InteractiveLoginAuthentication(tenant_id='9c8b18d3-0d09-4525-a546-341d38f12190')

ws = Workspace(subscription_id='83b4b5c6-51ae-4d5a-a7cf-63d20ffc2754',
               resource_group='MLproduct23',
               workspace_name='ltcml23',
               auth=auth)
'''

auth = InteractiveLoginAuthentication(tenant_id='839a5dd3-4df0-43b2-a507-3c1808d97dee')

ws = Workspace(subscription_id='8ea78212-550b-4294-97b7-1e432540fe46',
               resource_group='ML4',
               workspace_name='ltcftmlprd4',
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
from holoviews import dim, opts


def get_sweep_by(ctx, df_trainlog, key):
    df_trainlog = df_trainlog[df_trainlog[f'num_{key}'] > 1]
    if len(df_trainlog) > 0:
        df = pd.DataFrame({
            key: eval(str.encode(df_trainlog.iloc[0][f'sweep_{key}'])),
            'primary_metric': eval(str.encode(df_trainlog.iloc[0]['sweep_primary_metric']))
        })

        opts = dict(width=450, height=450, axiswise=True, show_legend=False)
        sc = hv.Scatter(df, key, 'primary_metric').options(**opts, jitter=0.5, size=5)
        bw = hv.BoxWhisker(df, key, 'primary_metric').options(**opts, box_fill_color='white')
        return (bw * sc).relabel(f'Generated {datetime.now().strftime("%c")}').opts(fontsize={'title': 10})
    else:
        opts = dict(width=450, height=450)
        return hv.Text(0.5, 0.5, 'No sweep jobs found').opts(**opts)

viz = get_sweep_by(None, df_trainlog, 'imputer')
viz

# %%

# %%

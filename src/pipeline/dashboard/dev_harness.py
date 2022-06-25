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
auth = InteractiveLoginAuthentication(tenant_id='839a5dd3-4df0-43b2-a507-3c1808d97dee')

ws = Workspace(subscription_id='8ea78212-550b-4294-97b7-1e432540fe46',
               resource_group='ML4',
               workspace_name='ltcftmlprd4',
               auth=auth)

ds = Datastore.get(ws, 'output')
df_runinfo = get_df(ds, 'Bank-Campaign/runinfo')
df_trainlog = get_df(ds, 'Bank-Campaign/trainlog')

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
from sklearn.metrics import roc_curve, auc, brier_score_loss

def get_correlation_plot(feature1, feature2):
    opts_scatter = dict(axiswise=True, jitter=0.2)
    opts_rasterize = dict(width=300, height=300)
    opts_spread = dict(axiswise=True, cmap=bp.Blues[256][::-1][64:], cnorm='eq_hist', padding=0.1)

    viz_scatter = hv.Scatter((feature1, feature2)).opts(**opts_scatter)
    viz_rasterize = rasterize(viz_scatter).opts(**opts_rasterize)
    viz_spread = spread(viz_rasterize, px=4, shape='square').opts(**opts_spread)

    return viz_spread


def get_histogram(f):
    opts_spikes = dict(line_alpha=0.4, spike_length=0.1)
    opts_rasterize = dict(width=300, height=300)
    opts_spread = dict(axiswise=True, cmap=bp.Reds[256][::-1][64:], cnorm='eq_hist')

    viz_spikes = hv.Spikes(f).opts(**opts_spikes)
    viz_rasterize = rasterize(viz_spikes).opts(**opts_rasterize)
    viz_spread = spread(viz_rasterize, px=4, shape='square').opts(**opts_spread)

    return viz_spread


def get_density_plots(df, df_trainlog):
    p = re.compile('feature_rank_([0-2])$')    
    features = df_trainlog.filter(regex=p, axis=1).iloc[0]

    # visual elements
    dict_grid = {
        f'{f1} x {f2}':get_correlation_plot(df[f1], df[f2]) if i1 != i2 else 
                       get_histogram(df[f1]) 
                       for (i1,f1) in enumerate(features) 
                       for (i2,f2) in enumerate(features)
    }

    grid = hv.NdLayout(dict_grid).cols(len(features))
    
    return grid

def get_viz():
    '''
        Begin visualization code
    '''
    viz = get_density_plots(dict_files['X_train'], df_trainlog)
    return viz
    

    '''
        End visualization code
    '''

viz = get_viz()
viz

# %%

# %%

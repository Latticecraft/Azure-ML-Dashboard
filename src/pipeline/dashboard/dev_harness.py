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

auth = InteractiveLoginAuthentication(tenant_id='9c8b18d3-0d09-4525-a546-341d38f12190')

ws = Workspace(subscription_id='83b4b5c6-51ae-4d5a-a7cf-63d20ffc2754',
               resource_group='MLproduct23',
               workspace_name='ltcml23',
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
from lazy_eval import get_theme

def get_samples_table(ctx, df_runinfo):
    p = re.compile('y_(train|valid|test)_([0-9])$')

    df = pd.DataFrame([{
        'value':v,
        'fold':p.match(k)[1],
        'class':p.match(k)[2]}
        for (k,v) in
        (df_runinfo
           .filter(regex=p, axis=1)
           .iloc[0]
           .to_dict()
           .items())])

    opts = dict(width=450, height=450, axiswise=True)
    bars_classdist = hv.Bars(df, ['fold','class'], 'value').opts(**opts)
    return bars_classdist.relabel(f'Generated {datetime.now().strftime("%c")}').opts(fontsize={'title': 10})


def get_label_distributions(ctx, dict_files):
    opts_dist = dict(filled=False, line_color=hv.Cycle())
    opts_overlay = dict(width=450, height=450)

    dists = {k: hv.Distribution(dict_files[k][ctx['label']]).opts(**opts_dist) for k in ['y_train', 'y_valid', 'y_test']}
    overlay = hv.NdOverlay(dists).opts(**opts_overlay)

    return overlay.relabel(f'Generated {datetime.now().strftime("%c")}').opts(fontsize={'title': 10})


def get_dtypes(ctx, df):
    p = re.compile('dtypes\.(predfs|postdfs)\.(bool|int64|float64)$')

    df_samples = df[[c for c in df.columns if p.match(c)]].reset_index(drop=True).to_dict()
    df3 = pd.DataFrame()
    for (k,v) in df_samples.items():
        m = p.match(k)
        df3 = df3.append({ 
            'value': df_samples[k][0],
            'stage': m[1],
            'dtype': m[2]
        }, ignore_index=True)

    data2 = hv.Dataset(df3, 
        [
            ('stage', 'Stage'), 
            ('dtype', 'Datatype')
        ], 
        [
            ('value', 'Value')
        ]
    )

    opts = dict(width=450, height=450, axiswise=True)
    bars_dtypes = hv.Bars(data2, ['stage','dtype'], 'value').opts(**opts)
    return bars_dtypes.relabel(f'Generated {datetime.now().strftime("%c")}').opts(fontsize={'title': 10})


def get_bivariate(ctx, df, df_trainlog):
    p = re.compile('feature_rank_([0-2])$')
    features = df_trainlog.filter(regex=p, axis=1).iloc[0]

    # visual elements
    dict_grid = {f'{f1} x {f2}': hv.HexTiles((df[f1], df[f2])).opts(xlabel=f1, ylabel=f2) for (i,f1) in enumerate(features) for (j,f2) in enumerate(features) if i < j}
    grid = hv.NdLayout(dict_grid).cols(3).relabel(f'Generated {datetime.now().strftime("%c")}').opts(fontsize={'title': 10})

    return grid.opts(
        opts.HexTiles(title='', scale=(dim('Count').norm()*0.5)+0.3, min_count=0, colorbar=False, padding=0.2, axiswise=True, framewise=True, shared_axes=False)
    )


def hook(plot, element):
    plot.state.title.text = 'Scatter Plot'

#viz_dtypes = get_dtypes(None, df_runinfo)
#viz_dtypes
theme = Theme(
    json={
'attrs' : {
    'Title':{
        'text': f'Generated {datetime.now().strftime("%c")}',
        'text_color': "white",
        'text_font_size': '6pt'
    },
    'Figure' : {
        'background_fill_color': '#414141',
        'border_fill_color': '#414141',
        'outline_line_color': '#444444',
    },
    'Grid': {
        'grid_line_dash': [6, 4],
        'grid_line_alpha': .3,
    },

    'Axis': {
        'major_label_text_color': 'white',
        'axis_label_text_color': 'white',
        'major_tick_line_color': 'white',
        'minor_tick_line_color': 'white',
        'axis_line_color': "white"
    },
    'Legend':{
        'background_fill_color': 'black',
        'background_fill_alpha': 0.5,
        'location' : "center_right",
        'label_text_color': "white"
    }
}
})

hv.renderer('bokeh').theme = theme

viz_biv = get_bivariate(None, X_train, df_trainlog)
viz_biv

# %%

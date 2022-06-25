#%%
import os, argparse
import re
import joblib
import numpy as np
import pandas as pd
import holoviews as hv
import panel as pn
import bokeh.palettes as bp
import glob
import shutil
import mlflow

from datetime import datetime, timedelta
from azureml.core import Run
from distutils.dir_util import copy_tree
from holoviews.operation.datashader import spread, rasterize
from pathlib import Path


hv.extension('bokeh')
pn.extension()


def get_samples_table(df_runinfo):
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
    return bars_classdist

def get_dtypes(df):
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
    bars_dtypes = hv.Bars(data2, ['stage','dtype'], 'value').options(**opts)
    return bars_dtypes


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


def get_df(path):
    df_all = pd.DataFrame()
    deltas = glob.glob(path+"/*")
    for d in deltas:
        print('adding {}'.format(d))
        df_delta = pd.read_csv((Path(path) / d), parse_dates=['runDate'])
        df_all = pd.concat([df_all, df_delta], ignore_index=True)

    df_all['runDate'] = pd.to_datetime(df_all['runDate'])
    return df_all.sort_values('runDate', ascending=False)


# define functions 
def main(ctx):
    df_runinfo = get_df(ctx['args'].runinfo)
    df_trainlog = get_df(ctx['args'].trainlog)

    runId = df_trainlog.sort_values('runDate', ascending=False).iloc[0]['sweep_get_best_model_runid']
    
    run = Run.get(workspace=ctx['run'].experiment.workspace, run_id=runId)
    run.download_file('outputs/model.pkl', output_file_path='data/model.pkl')
    run.download_file('outputs/datasets.pkl', output_file_path='data/datasets.pkl')

    model = joblib.load('data/model.pkl')
    dict_files = pd.read_pickle('data/datasets.pkl')

    viz_samples = get_samples_table(df_runinfo)
    viz_dtypes = get_dtypes(df_runinfo)
    viz_corr = get_density_plots(dict_files['X_train'], df_trainlog)

    viz_samples_dtypes = viz_samples + viz_dtypes

    hv.save(viz_samples, f'outputs/samples.html')
    hv.save(viz_dtypes, f'outputs/dtypes.html')
    hv.save(viz_samples_dtypes, f'outputs/samples_dtypes.html')
    hv.save(viz_corr, f'outputs/corr.html')
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
        'project': tags['project']
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
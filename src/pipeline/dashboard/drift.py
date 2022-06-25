#%%
import os, argparse
import re
import numpy as np
import pandas as pd
import holoviews as hv
import panel as pn
import bokeh.palettes as bp
import glob
import shutil
import mlflow

from azureml.core import Run
from datetime import datetime, timedelta
from distutils.dir_util import copy_tree
from pathlib import Path


hv.extension('bokeh')
pn.extension()


def get_datadrift(df):
    p = re.compile('^stats\.([a-zA-Z]*)_mean')

    df = (pd.DataFrame([{
        'value':list(v.values()),
        'value_mrr':list(v.values())[-1],
        'value_median':np.median(list(v.values())[:-1]),
        'value_std':np.std(list(v.values())[:-1]) or 1,
        'feature':p.match(k)[1],
        'alpha':list(np.repeat([0],len(list(v.values()))-1))+[1.0],
        'marker':list(np.repeat(['circle'],len(list(v.values()))-1))+['star'],
        'size':np.linspace(5,10,len(list(v.values()))),
        'color':list(np.repeat(['orange'],len(list(v.values()))-1))+['red']}
        for (k,v) in
        (df.set_index('runId')
           .filter(regex=p, axis=1)
           .to_dict()
           .items())])
        .explode(['value','alpha','marker','size','color'])
        .assign(
            value=lambda x:(x['value']-x['value_median'])/x['value_std'],
            divergence_mrr=lambda x:np.abs((x['value_mrr']-x['value_median'])/x['value_std'])
        )
        .sort_values('divergence_mrr', ascending=False))

    # determine sort order
    largest_divergences = list(df.where(lambda x: x['color'] == 'red')
            .filter(['feature','divergence_mrr'], axis=1)
            .sort_values(by='divergence_mrr', ascending=False)
            ['feature']
            .head(5))

    opts = dict(width=900, height=500)
    ds = hv.Dataset(df, kdims=['feature'], vdims=['value','alpha','marker','size','color'])
    sc = hv.Scatter(ds).options(**opts)
    bw = hv.BoxWhisker(ds, ['feature'], 'value').options(**opts)

    return bw[largest_divergences] * sc[largest_divergences]


def get_modeldrift(df):
    # data munge
    p = re.compile('(weighted avg_f1-score)|(runDate)')

    df = (df.set_index('runId')
            .filter(regex=p, axis=1)
            .assign(runDate=lambda x: x['runDate'].dt.date))

    # visual elements
    opts = dict(width=900, height=500)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=14)

    overlay = hv.NdOverlay({
            g: hv.Scatter(df[df['runDate'] == g], 'runDate', 'weighted avg_f1-score') for i,g in enumerate(df['runDate'])
        }).redim.values(runDate=[start_date, end_date]).options(**opts)

    errorbars = hv.ErrorBars([(k, el.reduce(function=np.mean), el.reduce(function=np.std)) for k, el in overlay.items()])

    curve = hv.Curve(errorbars)

    return errorbars * overlay * curve


def get_df(path):
    df_all = pd.DataFrame()
    deltas = glob.glob(path+"/*")
    for d in deltas:
        print('adding {}'.format(d))
        df_delta = pd.read_csv((Path(path) / d), parse_dates=['runDate'])
        df_all = pd.concat([df_all, df_delta], ignore_index=True)

    return df_all


# define functions 
def main(ctx):
    df_runinfo = get_df(ctx['args'].runinfo)
    df_trainlog = get_df(ctx['args'].trainlog)

    viz_datadrift = get_datadrift(df_runinfo)
    viz_modeldrift = get_modeldrift(df_trainlog)

    hv.save(viz_datadrift, f'outputs/datadrift.html')
    hv.save(viz_modeldrift, f'outputs/modeldrift.html')
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
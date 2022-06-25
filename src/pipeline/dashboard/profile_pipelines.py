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
from distutils.dir_util import copy_tree
from datetime import datetime, timedelta
from pathlib import Path


hv.extension('bokeh')
pn.extension()


def get_run_history(df_runinfo, df_trainlog):
    # munge
    end_date = datetime.now().date()
    start_date = (end_date - timedelta(days=14))

    df1 = (df_runinfo.assign(runDate=lambda x:x['runDate'].dt.date).groupby('runDate')['runId'].count())
    df2 = (df_trainlog.assign(runDate=lambda x:x['runDate'].dt.date).groupby('runDate')['runId'].count())
    df = pd.DataFrame({'featurizeCount':df1, 'trainCount':df2}).fillna(0).loc[start_date:end_date]

    # visual elements
    opts = dict(width=900, height=500)
    
    overlay = hv.Overlay([hv.Area((list(df.index), df.iloc[:,i])) for i in np.arange(2)])
    stack = hv.Area.stack(overlay).opts(**opts)

    return stack


def get_time_profile(df):
    p = re.compile('^JOB_([a-zA-Z_]*)\.([a-zA-Z_]*)')

    df2 = (pd.DataFrame([{
        'value_mrr':list(v.values())[-1],
        'job':p.match(k)[1],
        'attr':p.match(k)[2]}
        for (k,v) in
        (df.set_index('runId')
           .filter(regex=p, axis=1)
           .to_dict()
           .items()
        )])
        .set_index(['job', 'attr'])
        .unstack()
        .reset_index()
        .set_axis(['job', 'duration', 'isreused', 'startTime'], axis=1)
        .sort_values('startTime', ascending=True)
        .pipe(lambda x: x.loc[x['isreused'] == False])
    )

    opts = dict(width=900, height=500, xrotation=45)
    bars = hv.Area(df2, kdims=['job'], vdims=['duration']).options(**opts)

    return bars


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

    viz_runhistory = get_run_history(df_runinfo, df_trainlog)
    viz_time = get_time_profile(df_runinfo)

    os.makedirs("outputs", exist_ok=True)
    hv.save(viz_runhistory, f'outputs/runhistory.html')
    hv.save(viz_time, f'outputs/profile.html')
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
    parser.add_argument('--label', type=str)
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
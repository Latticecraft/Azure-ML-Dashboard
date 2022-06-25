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

from datetime import datetime, timedelta
from azureml.core import Run
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


def get_df(path):
    df_all = pd.DataFrame()
    deltas = glob.glob(path+"/*")
    for d in deltas:
        print('adding {}'.format(d))
        df_delta = pd.read_csv((Path(path) / d), parse_dates=['runDate'])
        df_all = pd.concat([df_all, df_delta], ignore_index=True)

    return df_all

def copy_html(source, destination, project_name, ts, filename):
    shutil.copyfile(source, Path(destination)/f'{project_name}-{ts}-{filename}.html')
    shutil.copyfile(source, Path(destination)/f'{project_name}-latest-{filename}.html')

# define functions 
def main(ctx):
    props = ctx['run'].parent.get_properties()
    project_name = props['projectname'].lower()

    df_runinfo = get_df(ctx['args'].runinfo)
    df_trainlog = get_df(ctx['args'].trainlog)

    viz_runhistory = get_run_history(df_runinfo, df_trainlog)
    viz_samples = get_samples_table(df_runinfo)
    viz_dtypes = get_dtypes(df_runinfo)
    viz_time = get_time_profile(df_runinfo)
    viz_datadrift = get_datadrift(df_runinfo)

    viz_samples_dtypes = viz_samples + viz_dtypes

    os.makedirs("outputs", exist_ok=True)
    hv.save(viz_runhistory, f'outputs/runhistory.html')
    hv.save(viz_samples, f'outputs/samples.html')
    hv.save(viz_dtypes, f'outputs/dtypes.html')
    hv.save(viz_samples_dtypes, f'outputs/samples_dtypes.html')
    hv.save(viz_time, f'outputs/profile.html')
    hv.save(viz_datadrift, f'outputs/datadrift.html')

    # register dataset
    now = datetime.now()
    ts = now.strftime('%m%d%y')

    copy_html(Path('outputs') / 'runhistory.html', ctx['args'].transformed_data, project_name, ts, 'runhistory')
    copy_html(Path('outputs') / 'samples.html', ctx['args'].transformed_data, project_name, ts, 'samples')
    copy_html(Path('outputs') / 'dtypes.html', ctx['args'].transformed_data, project_name, ts, 'dtypes')
    copy_html(Path('outputs') / 'samples_dtypes.html', ctx['args'].transformed_data, project_name, ts, 'samples_dtypes')
    copy_html(Path('outputs') / 'profile.html', ctx['args'].transformed_data, project_name, ts, 'profile')
    copy_html(Path('outputs') / 'datadrift.html', ctx['args'].transformed_data, project_name, ts, 'datadrift')


def start(args):
    os.makedirs("outputs", exist_ok=True)
    mlflow.start_run()
    client = mlflow.tracking.MlflowClient()
    return {
        'args': args,
        'run': Run.get_context(),
        'data': client.get_run(mlflow.active_run().info.run_id).data
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
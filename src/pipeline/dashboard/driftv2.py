#%%
import argparse 
import holoviews as hv
import numpy as np
import os
import pandas as pd
import re
import sys

from datetime import datetime, timedelta

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from commonv2 import *

hv.extension('bokeh')


#%%
def data_drift(func):
    def inner(**kwargs):
        df = kwargs['data_layer.runinfo']

        df['runDate'] = pd.to_datetime(df['runDate'])
        df['runDate'] = df['runDate'].dt.strftime('%Y-%m-%d')

        p = re.compile('^stats\.([a-zA-Z0-9_ ]*)_mean')

        print(f'len df_0: {len(df)}')

        df_1 = (df.set_index('runDate')
                .filter(regex=p, axis=1)
                .groupby('runDate')
                .mean()
                .reset_index(drop=False)
                .sort_values('runDate', ascending=False))

        print(f'len df_1: {len(df_1)}')

        df_all = pd.DataFrame()
        for rd in df_1['runDate'].head(np.min([len(df_1)-2, 10])):
            df_2 = (pd.DataFrame([{
                'feature': p.match(k)[1],
                'divergence_mrr': np.abs(v[-1]-np.mean(v[:-1])/(np.std(v[:-1]) or 1))
            } for (k,v) in df_1[df_1['runDate'] < rd].drop(['runDate'], axis=1).dropna(thresh=2, axis=1).to_dict(orient='list').items()])
            .sort_values('divergence_mrr', ascending=False)
            .assign(runDate=rd))
            df_all = pd.concat([df_all, df_2], ignore_index=True)

        print(f'len df_all: {len(df_all)}')

        if len(np.unique(df_all['runDate'])) > 5:
            # get top 10 divergent features
            top_divergences = (df_all
                .sort_values('divergence_mrr', ascending=False)
                .drop_duplicates(subset=['feature'])
                .head(10))['feature']

            # filter out features not in above list
            df_all = (df_all
                .sort_values('runDate', ascending=False)
                .merge(top_divergences, on='feature', how='inner'))

            x = np.array(df_all['runDate'])
            y = np.array(df_all['feature'])
            v = np.array(df_all['divergence_mrr'])
            hm = hv.HeatMap((x,y,v))

            kwargs['plot'] = hm
            kwargs['plot_name'] = 'datadrift'

        return func(**kwargs)

    return inner


def model_drift(func):
    def inner(**kwargs):
        df = kwargs['data_layer.trainlog']

        p = re.compile(f'({kwargs["primary_metric"]})|(runDate)')

        df = (df.set_index('runId')
                .filter(regex=p, axis=1)
                .assign(runDate=lambda x: x['runDate'].dt.date))

        end_date = datetime.now()
        start_date = end_date - timedelta(days=14)

        x = np.array(df['runDate'])
        y = np.array(df[kwargs['primary_metric']])
        curve = hv.Curve((x,y)).redim.values(runDate=[start_date, end_date])

        kwargs['plot'] = curve
        kwargs['plot_name'] = 'modeldrift'

        return func(**kwargs)

    return inner


def config(func):
    def inner(**kwargs):
        if 'plot' in kwargs:
            kwargs['plot'] = kwargs['plot'].opts(
                width=900, height=500
            )

        return func(**kwargs)

    return inner


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()

    # add arguments
    parser.add_argument('--datasets-pkl', type=str, default='data')
    parser.add_argument('--runinfo', type=str, default='data/runinfo')
    parser.add_argument('--trainlog', type=str, default='data/trainlog')
    parser.add_argument('--transformed-data', type=str)
    
    # parse args
    args = parser.parse_args()

    # return args
    return args


#%%
# run script
if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)

    try:
        args
    except NameError:
        args = vars(parse_args())

    
    viz_ddrift = (
        context(
            data(
                data_drift(
                    config(
                        export
                    )
                )
            )
        )(**{
            **args, 
            'include_runinfo': True, 
            'include_trainlog': True
        }))
    

    viz_mdrift = (
        context(
            data(
                model_drift(
                    config(
                        export
                    )
                )
            )
        )(**{
            **args, 
            'include_runinfo': True, 
            'include_trainlog': True
        }))

# %%

#%%
import argparse 
import holoviews as hv
import numpy as np
import os
import pandas as pd
import re
import sys

from bokeh.models.formatters import DatetimeTickFormatter

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from commonv2 import *

hv.extension('bokeh')


#%%
def data_drift(func):
    def inner(**kwargs):
        df = kwargs['data_layer.runinfo']

        runDates = df.sort_values('runDate', ascending=False)['runDate'].head(5)

        p = re.compile('^stats\.([a-zA-Z0-9_ ]*)_mean|runDate')

        print(f'len df_0: {len(df)}')

        df_1 = (df.set_index('runId')
                .filter(regex=p, axis=1))

        print(f'len df_1: {len(df_1)}')

        df_2 = (pd.DataFrame([{
            'feature': p.match(k)[1],
            'divergence_mrr': np.abs(v[-1]-np.mean(v[:-1])/(np.std(v[:-1]) or 1))
        } for (k,v) in df_1[df_1['runDate'] < np.max(runDates)].drop(['runDate'], axis=1).to_dict(orient='list').items()])
        .sort_values('divergence_mrr', ascending=False)
        .head(5))

        print(f'len df_2: {len(df_2)}')

        opts = dict(width=900, height=500)

        x = np.array(df_2['runDate'])
        y = np.array(df_2['feature'])
        v = np.array(df_2['divergence_mrr'])
        hm = hv.HeatMap((x,y,v)).opts(**opts)

        kwargs['plot'] = hm
        kwargs['plot_name'] = 'datadrift'

        return func(**kwargs)

    return inner


def config(func):
    def inner(**kwargs):
        return func(**kwargs)
        '''.opts(
            opts.Scatter(width=900, height=500, jitter=0.5, size=5),
            opts.BoxWhisker(width=900, height=500, box_fill_color='white', whisker_color='gray')
        )'''

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
if __name__ == "__main__":
    os.makedirs("outputs", exist_ok=True)

    try:
        args
    except NameError:
        args = vars(parse_args())

    viz_ddrift = (
        context(
            data(
                data_drift(
                    export
                )
            )
        )(**{
            **args, 
            'include_runinfo': True, 
            'include_trainlog': True
        }))

# %%

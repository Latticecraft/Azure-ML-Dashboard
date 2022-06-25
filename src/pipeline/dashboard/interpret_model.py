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
from holoviews import dim, opts
from holoviews.operation.datashader import spread, rasterize
from interpret.ext.blackbox import TabularExplainer
from pathlib import Path
from sklearn.metrics import auc, confusion_matrix, classification_report, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve


hv.extension('bokeh')
pn.extension()


def get_shap(explainer, dict_files, feature):
    # get metrics
    explanations = explainer.explain_local(dict_files['X_test'])
    ranked_names = explanations.get_ranked_local_names()
    ranked_vals = explanations.get_ranked_local_values()

    # munge
    df_tmp = pd.DataFrame(ranked_names)
    df_names = pd.DataFrame(df_tmp.iloc[1].values.tolist(), index=df_tmp.iloc[1].index)

    df_tmp = pd.DataFrame(ranked_vals)
    df_vals = pd.DataFrame(df_tmp.iloc[1].values.tolist(), index=df_tmp.iloc[1].index)

    feature_map = {
        feature: [(x == feature).argmax() for (i,x) in df_names.iterrows()]
    }

    X = {
        feature: [df_vals.iloc[i][x] for (i,x) in enumerate(feature_map[feature])]
    }

    y = {
        feature: [x for x in dict_files['X_test'][feature]]
    }

    df_scatter = pd.DataFrame({'x': X[feature], 'y': y[feature]})

    # visualize
    opts_scatter = dict(jitter=0.2)    
    opts_shaded = dict(axiswise=True, cmap=bp.Blues[256][::-1][64:], cnorm='eq_hist', padding=0.1)
    opts_curve = dict(axiswise=True, line_dash='dashed', color='black')
    overlay_opts = dict(axiswise=True)

    scatter = hv.Scatter(df_scatter).opts(**opts_scatter)
    shaded = spread(rasterize(scatter), px=4, shape='circle').opts(**opts_shaded)
    curve = hv.Curve([[0,np.min(y[feature])], [0,np.max(y[feature])]]).opts(**opts_curve)

    return (shaded * curve).opts(**overlay_opts)


def get_shapgrid(explainer, dict_files):
    explanations = explainer.explain_global(dict_files['X_test'])
    dict = explanations.get_feature_importance_dict()
    top_features = list(pd.DataFrame(dict, index=[1]).T.head(9).index)

    dict_grid = {f:get_shap(explainer, dict_files, f) for (i,f) in enumerate(top_features)}
    grid = hv.NdLayout(dict_grid).cols(3)
    return grid


def get_feature_importances(df_trainlog):
    # munge
    p = re.compile('runDate|feature_rank_([0-9]{1,2})$')    
    df = (df_trainlog.filter(regex=p, axis=1).T
            .pipe(lambda x: x.set_axis(x.iloc[0], axis=1))
            .iloc[1:]
            .assign(y=lambda x: np.abs(x.index.to_series().str.extract(p).astype('int')-len(x)))
            .reset_index(drop=True)
            .pipe(lambda x: pd.melt(x, id_vars=['y'], value_vars=x.columns))
            .rename({'runDate':'X', 'value':'feature'}, axis=1)
            .assign(feature=lambda x: x['feature'].apply(str),
                weight=lambda x: np.ones(len(x))))[['feature', 'X', 'y', 'weight']]

    end_date = df_trainlog['runDate'].max()
    start_date = df_trainlog['runDate'].min()
    if (end_date - start_date).days > 14:
        start_date = end_date - timedelta(days=14)

    # visual elements
    opts_curve = dict(width=900, height=500, yaxis='right')

    curves = [hv.Curve(df.loc[df['feature']==f], kdims=['X'], vdims=['y', 'weight'], label=f).redim.values(y=np.arange(df['y'].max(), df['y'].max()-5, -1)).opts(**opts_curve) for f in np.unique(df['feature'])]
    yticks = [(abs(i-sum(df['X'] == end_date)),x) for i,x in enumerate(df.loc[df['X'] == end_date, 'feature'])]

    overlay = hv.Overlay(curves).opts(
        width=900,
        height=500, 
        xlabel='Time', 
        ylabel='Importance', 
        title='Change in top features over runs',
        show_legend=False
    ).opts(yticks=yticks)

    return overlay


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
    df_trainlog = get_df(ctx['args'].trainlog)

    runId = df_trainlog.sort_values('runDate', ascending=False).iloc[0]['sweep_get_best_model_runid']

    run = Run.get(workspace=ctx['run'].experiment.workspace, run_id=runId)
    run.download_file('outputs/model.pkl', output_file_path='data/model.pkl')
    run.download_file('outputs/datasets.pkl', output_file_path='data/datasets.pkl')

    model = joblib.load('data/model.pkl')
    dict_files = pd.read_pickle('data/datasets.pkl')

    explainer = TabularExplainer(model, dict_files['X_train'])

    viz_shap = get_shapgrid(explainer, dict_files)
    viz_feature = get_feature_importances(df_trainlog)

    hv.save(viz_shap, f'outputs/shap.html')
    hv.save(viz_feature, f'outputs/features.html')
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
import argparse, joblib, json, mlflow, os, re, sys 
import numpy as np
import pandas as pd
import holoviews as hv

from azureml.core import Run
from bokeh.io import export_png
from datetime import timedelta
from distutils.dir_util import copy_tree
from holoviews import dim, opts
from interpret.ext.blackbox import TabularExplainer

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from common import LazyEval, get_df, get_webdriver

hv.extension('bokeh')


def get_shap(explainer, files, feature):
    # get metrics
    explanations = explainer.explain_local(files['X_test'])
    ranked_names = explanations.get_ranked_local_names()
    ranked_vals = explanations.get_ranked_local_values()

    # munge
    if len(np.shape(ranked_names)) == 2: # Regression
        df_tmp = pd.DataFrame(ranked_names)
        df_names = pd.DataFrame(df_tmp.values.tolist(), index=df_tmp.index)

        df_tmp = pd.DataFrame(ranked_vals)
        df_vals = pd.DataFrame(df_tmp.values.tolist(), index=df_tmp.index)
    
    else: # Classification
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
        feature: [x for x in files['X_test'][feature]]
    }

    df_hex = pd.DataFrame({'x': X[feature], 'y': y[feature]})

    # visualize
    hex = hv.HexTiles(df_hex)
    
    return hex.opts(
        opts.HexTiles(min_count=1, width=300, height=300, scale=(dim('Count').norm()*0.5)+0.3, colorbar=False, padding=0.2, axiswise=True, framewise=True, shared_axes=False),
    )

def get_shapgrid(explainer, files):
    explanations = explainer.explain_global(files['X_test'])
    dict_fi = explanations.get_feature_importance_dict()
    top_features = list(pd.DataFrame(dict_fi, index=[1]).T.head(9).index)

    dict_grid = {f:get_shap(explainer, files, f) for (i,f) in enumerate(top_features)}
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


# define functions 
def main(ctx):
    df_trainlog = get_df(ctx['args'].trainlog)

    runId = df_trainlog.sort_values('runDate', ascending=False).iloc[0]['sweep_get_best_model_runid']

    run = Run.get(workspace=ctx['run'].experiment.workspace, run_id=runId)
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
    X_test, y_test = data.get('test', imputer, balancer)

    dict_new = {
        'X_train': X_train,
        'y_train': y_train,
        'X_test': X_test,
        'y_test': y_test
    }

    explainer = TabularExplainer(model, dict_new['X_train'])

    webdriver = get_webdriver()

    viz_shap = get_shapgrid(explainer, dict_new)
    viz_feature = get_feature_importances(df_trainlog)

    hv.save(viz_shap, f'outputs/shap.html')
    export_png(hv.render(viz_shap), filename= 'outputs/shap.png', webdriver=webdriver)

    hv.save(viz_feature, f'outputs/features.html')
    export_png(hv.render(viz_feature), filename= 'outputs/features.png', webdriver=webdriver)

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
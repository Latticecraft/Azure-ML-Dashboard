import argparse, joblib, json, mlflow, os, re, sys 
import pandas as pd
import holoviews as hv

from azureml.core import Run
from bokeh.io import export_png
from distutils.dir_util import copy_tree
from holoviews import dim, opts

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from common import LazyEval, get_df, get_webdriver

hv.extension('bokeh')


def get_samples_table(ctx, df_runinfo):
    p = re.compile('y_(train|test)_([0-9])$')

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


def get_label_distributions(ctx, dict_files):
    opts_dist = dict(filled=False, line_color=hv.Cycle())
    opts_overlay = dict(width=450, height=450)

    dists = {k: hv.Distribution(dict_files[k][ctx['label']]).opts(**opts_dist) for k in ['y_train', 'y_test']}
    overlay = hv.NdOverlay(dists).opts(**opts_overlay)

    return overlay


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
    return bars_dtypes


def get_bivariate(ctx, df, df_trainlog):
    p = re.compile('feature_rank_([0-2])$')
    features = df_trainlog.filter(regex=p, axis=1).iloc[0]

    # visual elements
    dict_grid = {f'{f1} x {f2}': hv.HexTiles((df[f1], df[f2])).opts(xlabel=f1, ylabel=f2) for (i,f1) in enumerate(features) for (j,f2) in enumerate(features) if i < j}
    grid = hv.NdLayout(dict_grid).cols(3)

    return grid.opts(
        opts.HexTiles(title='', scale=(dim('Count').norm()*0.5)+0.3, min_count=1, colorbar=False, padding=0.2, axiswise=True, framewise=True, shared_axes=False)
    )


# define functions 
def main(ctx):
    df_runinfo = get_df(ctx['args'].runinfo)
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

    webdriver = get_webdriver()

    viz_dtypes = get_dtypes(ctx, df_runinfo)
    viz_biv = get_bivariate(ctx, X_train, df_trainlog)

    hv.save(viz_dtypes, f'outputs/dtypes.html')
    export_png(hv.render(viz_dtypes), filename= 'outputs/dtypes.png', webdriver=webdriver)

    hv.save(viz_biv, f'outputs/hextiles_top3.html')
    export_png(hv.render(viz_biv), filename= 'outputs/hextiles_top3.png', webdriver=webdriver)    
    
    if ctx['type'] != 'Regression':
        viz_samples = get_samples_table(ctx, df_runinfo)
        viz_samples_dtypes = viz_samples + viz_dtypes

        hv.save(viz_samples, f'outputs/samples.html')
        export_png(hv.render(viz_samples), filename= 'outputs/samples.png', webdriver=webdriver)  

        hv.save(viz_samples_dtypes, f'outputs/samples_dtypes.html')
        export_png(hv.render(viz_samples_dtypes), filename= 'outputs/samples_dtypes.png', webdriver=webdriver)  


    else:
        viz_label_dist = get_label_distributions(ctx, dict_new)
        viz_label_distributions_dtypes = viz_label_dist + viz_dtypes

        hv.save(viz_label_dist, f'outputs/label_distributions.html')
        export_png(hv.render(viz_label_dist), filename= 'outputs/label_distributions.png', webdriver=webdriver)

        hv.save(viz_label_distributions_dtypes, f'outputs/label_distributions_dtypes.html')
        export_png(hv.render(viz_label_distributions_dtypes), filename= 'outputs/label_distributions_dtypes.png', webdriver=webdriver)

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
        'project': tags['project'],
        'type': tags['type'],
        'label': tags['label']
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
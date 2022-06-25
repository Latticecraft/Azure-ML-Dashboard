#%%
import os, argparse
import re
import joblib
import numpy as np
import pandas as pd
import holoviews as hv
import panel as pn
import glob
import shutil
import bokeh.palettes as bp
import mlflow

from azureml.core import Run
from datetime import datetime, timedelta
from distutils.dir_util import copy_tree
from holoviews import dim, opts
from holoviews.operation.datashader import spread, rasterize
from interpret.ext.blackbox import TabularExplainer
from pathlib import Path
from sklearn.metrics import auc, confusion_matrix, classification_report, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve


hv.extension('bokeh')
pn.extension()


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


def get_confusion_matrix(model, dict_files, label):
    # generate metrics
    yhat_proba = [x[1] for x in model.predict_proba(dict_files['X_test'])]
    yhat = [1 if x >= 0.5 else 0 for x in yhat_proba]

    tn, fp, fn, tp = confusion_matrix(dict_files['y_test'], yhat).ravel()
    total = tn + fp + fn + tp

    report = classification_report(dict_files['y_test'][label].ravel(), yhat, output_dict=True)
    precision = round(report['macro avg']['precision'], 4)
    recall = round(report['macro avg']['recall'], 4)
    f1_macro = round(report['macro avg']['f1-score'], 4)
    f1_weighted = round(report['weighted avg']['f1-score'], 4)
    accuracy = round(report['accuracy'], 4)

    # visual elements
    confmatrix = (
        (hv.HeatMap([(0, 0, fp/total), (0, 1, tp/total), (1, 0, tn/total), (1, 1, fn/total)]).opts(width=600, height=500) * 
            hv.Text(0,1.125,'TP') *
            hv.Text(0,1,str(tp)) *
            hv.Text(1,1.125,'FN') *
            hv.Text(1,1,str(fn)) *
            hv.Text(0,.125,'FP') *
            hv.Text(0, 0,str(fp)) *
            hv.Text(1,0.125,'TN') *
            hv.Text(1,0,str(tn))) + 
        (hv.HeatMap([(0, 5, precision), (0, 4, recall), (0, 3, f1_macro), (0, 2, f1_weighted), (0, 1, accuracy)]).redim.range(z=(0, 1)).opts(width=300, height=500) *
            hv.Text(0,5.125,'Precision (Macro Avg)') *
            hv.Text(0,4.875,str(precision)) *
            hv.Text(0,4.125,'Recall (Macro Avg)') *
            hv.Text(0,3.875,str(recall)) *
            hv.Text(0,3.125,'F1 Score (Macro Avg)') *
            hv.Text(0,2.875,str(f1_macro)) *
            hv.Text(0,2.125,'F1 Score (Weighted Avg)') *
            hv.Text(0,1.875,str(f1_weighted)) *
            hv.Text(0,1.125,'Accuracy') *
            hv.Text(0,0.875,str(round(accuracy, 2)))).opts(axiswise=True)
        ).opts(
            opts.HeatMap(axiswise=True, cmap='coolwarm'), 
            opts.Text(axiswise=True)
        )

    return confmatrix


def get_reliability_curve(model, dict_files):
    # get metrics
    yhat_proba = [x[1] for x in model.predict_proba(dict_files['X_test'])]
    prob_true, prob_pred = calibration_curve(dict_files['y_test'], yhat_proba)
    brier_score = brier_score_loss(dict_files['y_test'], yhat_proba)

    # munge
    df = pd.DataFrame({
        'prob_true': prob_true,
        'prob_pred': prob_pred
    })

    df_labels = pd.DataFrame({'x1':[0.55], 'y1':[0.05], 'x2':[0.95], 'y2':[0.3]})

    # visual elements
    curve = (hv.Curve(df[['prob_pred', 'prob_true']]) *
             hv.Curve([[0,0], [1,1]]).opts(line_dash='dashed', color='black') *
             hv.Rectangles(df_labels, kdims=['x1', 'y1', 'x2', 'y2'], vdims=[]).redim.range(c=(1, 0)).opts(color='white') *
             hv.Text(0.75, 0.2, 'Brier Score') *
             hv.Text(0.75, 0.15, str(round(brier_score, 3))))

    opts = dict(width=450, height=450)
    return curve.opts(**opts)


def get_roc(model, dict_files):
    # get metrics
    yhat_proba = [x[1] for x in model.predict_proba(dict_files['X_test'])]
    fpr, tpr, thresholds = roc_curve(dict_files['y_test'], yhat_proba)
    area = auc(fpr, tpr)

    # munge
    df = pd.DataFrame({
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    })

    df_labels = pd.DataFrame({'x1':[0.55], 'y1':[0.05], 'x2':[0.95], 'y2':[0.3]})

    # visualize
    opts = dict(width=450, height=450)
    roc = (hv.Area(df[['fpr', 'tpr']]) *
           hv.Rectangles(df_labels, kdims=['x1', 'y1', 'x2', 'y2'], vdims=[]).opts(color='white') *
           hv.Text(0.75, 0.2, 'AUC') *
           hv.Text(0.75, 0.15, str(round(area, 3)))).opts(**opts)

    return roc


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


def get_sweep_by(df_trainlog, key):
    df = pd.DataFrame({
        'balancer': eval(str.encode(df_trainlog.iloc[0]['sweep_balancer'])),
        'imputer': eval(str.encode(df_trainlog.iloc[0]['sweep_imputer'])),
        'weighted avg_f1-score': eval(str.encode(df_trainlog.iloc[0]['sweep_weighted avg_f1-score']))
    })
    
    opts = dict(width=450, height=450)
    sc = hv.Scatter(df, [key], 'weighted avg_f1-score').options(**opts)
    bw = hv.BoxWhisker(df, [key], 'weighted avg_f1-score').options(**opts)
    return bw * sc


def get_df(path):
    df_all = pd.DataFrame()
    deltas = glob.glob(path+"/*")
    for d in deltas:
        print('adding {}'.format(d))
        df_delta = pd.read_csv((Path(path) / d), parse_dates=['runDate'])
        df_all = pd.concat([df_all, df_delta], ignore_index=True)

    return df_all.sort_values('runDate', ascending=False)


def copy_html(source, destination, project_name, ts, filename):
    shutil.copyfile(source, Path(destination)/f'{project_name}-{ts}-{filename}.html')
    shutil.copyfile(source, Path(destination)/f'{project_name}-latest-{filename}.html')


# define functions 
def main(ctx):
    props = ctx['run'].parent.get_properties()
    project_name = props['projectname'].lower()

    df_trainlog = get_df(ctx['args'].trainlog)

    runId = df_trainlog.sort_values('runDate', ascending=False).iloc[0]['sweep_get_best_model_runid']

    run = Run.get(workspace=ctx['run'].experiment.workspace, run_id=runId)
    run.download_file('outputs/model.pkl', output_file_path='data/model.pkl')
    run.download_file('outputs/datasets.pkl', output_file_path='data/datasets.pkl')

    model = joblib.load('data/model.pkl')
    dict_files = pd.read_pickle('data/datasets.pkl')

    explainer = TabularExplainer(model, dict_files['X_train'])

    modeldrift_viz = get_modeldrift(df_trainlog)
    confmatrix_viz = get_confusion_matrix(model, dict_files, ctx['args'].label)
    reliability_viz = get_reliability_curve(model, dict_files)
    roc_viz = get_roc(model, dict_files)
    shap_viz = get_shapgrid(explainer, dict_files)
    feature_viz = get_feature_importances(df_trainlog)
    viz_corr = get_density_plots(dict_files['X_train'], df_trainlog)
    viz_sweep = get_sweep_by(df_trainlog, 'balancer') + get_sweep_by(df_trainlog, 'imputer')

    roc_reliability = roc_viz + reliability_viz
    
    os.makedirs("outputs", exist_ok=True)
    hv.save(modeldrift_viz, f'outputs/modeldrift.html')
    hv.save(reliability_viz, f'outputs/reliability.html')
    hv.save(roc_viz, f'outputs/roc.html')
    hv.save(roc_reliability, f'outputs/roc_reliability.html')
    hv.save(shap_viz, f'outputs/shap.html')
    hv.save(confmatrix_viz, f'outputs/confmatrix.html')
    hv.save(feature_viz, f'outputs/features.html')
    hv.save(viz_corr, f'outputs/corr.html')
    hv.save(viz_sweep, f'outputs/sweep.html')

    # register dataset
    now = datetime.now()
    ts = now.strftime('%m%d%y')

    copy_html(Path('outputs') / 'modeldrift.html', ctx['args'].transformed_data, project_name, ts, 'modeldrift')
    copy_html(Path('outputs') / 'reliability.html', ctx['args'].transformed_data, project_name, ts, 'reliability')
    copy_html(Path('outputs') / 'roc.html', ctx['args'].transformed_data, project_name, ts, 'roc')
    copy_html(Path('outputs') / 'roc_reliability.html', ctx['args'].transformed_data, project_name, ts, 'roc_reliability')
    copy_html(Path('outputs') / 'shap.html', ctx['args'].transformed_data, project_name, ts, 'shap')
    copy_html(Path('outputs') / 'confmatrix.html', ctx['args'].transformed_data, project_name, ts, 'confmatrix')
    copy_html(Path('outputs') / 'features.html', ctx['args'].transformed_data, project_name, ts, 'features')
    copy_html(Path('outputs') / 'corr.html', ctx['args'].transformed_data, project_name, ts, 'corr')
    copy_html(Path('outputs') / 'sweep.html', ctx['args'].transformed_data, project_name, ts, 'sweep')


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
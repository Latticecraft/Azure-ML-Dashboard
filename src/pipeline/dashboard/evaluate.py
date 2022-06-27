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
from interpret.ext.blackbox import TabularExplainer
from pathlib import Path
from sklearn.metrics import auc, confusion_matrix, classification_report, roc_curve, brier_score_loss
from sklearn.calibration import calibration_curve


hv.extension('bokeh')
pn.extension()


def get_residuals_plot(ctx, model, dict_files):
    # get metrics
    yhat = model.predict(dict_files['X_test'])
    residuals = dict_files['y_test'][ctx['label']] - yhat

    # visualize
    opts = dict(width=450, height=450)
    dist = hv.Distribution(residuals).opts(**opts)

    return dist


def get_confusion_matrix(ctx, model, dict_files, label):
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


def get_reliability_curve(ctx, model, dict_files):
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


def get_roc(ctx, model, dict_files):
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


def get_sweep_by(ctx, df_trainlog, key):
    df = pd.DataFrame({
        'balancer': eval(str.encode(df_trainlog.iloc[0]['sweep_balancer'])),
        'imputer': eval(str.encode(df_trainlog.iloc[0]['sweep_imputer'])),
        'primary_metric': eval(str.encode(df_trainlog.iloc[0]['sweep_primary_metric']))
    })
    
    opts = dict(width=450, height=450)
    sc = hv.Scatter(df, [key], 'primary_metric').options(**opts)
    bw = hv.BoxWhisker(df, [key], 'primary_metric').options(**opts)
    return bw * sc


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
    df_trainlog = get_df(ctx['args'].trainlog)

    runId = df_trainlog.sort_values('runDate', ascending=False).iloc[0]['sweep_get_best_model_runid']

    run = Run.get(workspace=ctx['run'].experiment.workspace, run_id=runId)
    run.download_file('outputs/model.pkl', output_file_path='data/model.pkl')
    run.download_file('outputs/datasets.pkl', output_file_path='data/datasets.pkl')

    model = joblib.load('data/model.pkl')
    dict_files = pd.read_pickle('data/datasets.pkl')

    if ctx['type'] != 'Regression':
        viz_confmatrix = get_confusion_matrix(ctx, model, dict_files, ctx['label'])
        viz_reliability = get_reliability_curve(ctx, model, dict_files)
        viz_roc = get_roc(ctx, model, dict_files)
        roc_reliability = viz_roc + viz_reliability

        hv.save(viz_confmatrix, f'outputs/confmatrix.html')
        hv.save(viz_reliability, f'outputs/reliability.html')
        hv.save(viz_roc, f'outputs/roc.html')
        hv.save(roc_reliability, f'outputs/roc_reliability.html')

    else:
        viz_residuals = get_residuals_plot(ctx, model, dict_files)

        hv.save(viz_residuals, f'outputs/residuals.html')

    viz_sweep = get_sweep_by(ctx, df_trainlog, 'balancer') + \
                get_sweep_by(ctx, df_trainlog, 'imputer')

    hv.save(viz_sweep, f'outputs/sweep.html')

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
        'label': tags['label'],
        'primary_metric': tags['primary_metric']
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
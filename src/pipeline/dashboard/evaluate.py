#%%
import sys, os, argparse
import json
import joblib
import pandas as pd
import holoviews as hv
import panel as pn
import glob
import mlflow

from azureml.core import Run
from datetime import datetime
from distutils.dir_util import copy_tree
from holoviews import dim, opts
from pathlib import Path
from sklearn.metrics import auc, confusion_matrix, classification_report, roc_curve, brier_score_loss, explained_variance_score, mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.calibration import calibration_curve

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path)
from lazy_eval import LazyEval


hv.extension('bokeh')
pn.extension()


def get_residuals_plot(ctx, model, dict_files):
    # get metrics
    yhat = model.predict(dict_files['X_test'])
    residuals = dict_files['y_test'][ctx['label']] - yhat
    explained_variance = explained_variance_score(dict_files['y_test'], yhat)
    mae = mean_absolute_error(dict_files['y_test'], yhat)
    rmse = mean_squared_error(dict_files['y_test'], yhat, squared=False)
    r2 = r2_score(dict_files['y_test'], yhat)
    mape = mean_absolute_percentage_error(dict_files['y_test'], yhat)

    # visualize
    opts_dist = dict(width=450, height=450)
    opts_metrics = dict(width=450, height=450, xlim=(0,1), ylim=(0,1), xaxis='bare', yaxis='bare')

    dist = hv.Distribution(residuals).opts(**opts_dist)

    metrics = (hv.Text(0.3,0.9,'Explained Variance') *
        hv.Text(0.3,0.85,str(round(explained_variance, 3))) *
        hv.Text(0.3,0.7,'MAE') *
        hv.Text(0.3,0.65,str(round(mae, 3))) *
        hv.Text(0.3,0.5,'RMSE') *
        hv.Text(0.3,0.45,str(round(rmse, 3))) *
        hv.Text(0.3,0.3,'R^2') *
        hv.Text(0.3,0.25,str(round(r2, 3))) *
        hv.Text(0.3,0.1,'MAPE') *
        hv.Text(0.3,0.05,str(round(mape, 3)))).opts(**opts_metrics)
        
    overlay = dist + metrics

    return overlay


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
    df_trainlog = df_trainlog[df_trainlog[f'num_{key}'] > 1]
    if len(df_trainlog) > 0:
        df = pd.DataFrame({
            key: eval(str.encode(df_trainlog.iloc[0][f'sweep_{key}'])),
            'primary_metric': eval(str.encode(df_trainlog.iloc[0]['sweep_primary_metric']))
        })

        opts = dict(width=450, height=450, axiswise=True, show_legend=False)
        sc = hv.Scatter(df, key, 'primary_metric').options(**opts, jitter=0.5, size=5)
        bw = hv.BoxWhisker(df, key, 'primary_metric').options(**opts, box_fill_color='white')
        return (bw * sc).relabel(f'Generated {datetime.now().strftime("%c")}').opts(fontsize={'title': 10})
    else:
        opts = dict(width=450, height=450)
        return hv.Text(0.5, 0.5, 'No sweep jobs found').opts(**opts)


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
    run.download_file('outputs/best_run.json', output_file_path='data/best_run.json')

    with open('data/best_run.json', 'r') as f:
        best_run = json.load(f)
        imputer = best_run['imputer']
        balancer = best_run['balancer']

    model = joblib.load('data/model.pkl')
    dict_files = pd.read_pickle('data/datasets.pkl')
    data = LazyEval(dict_files)

    X_train, y_train = data.get('train', imputer, balancer)
    X_valid, y_valid = data.get('valid', imputer, balancer)
    X_test, y_test = data.get('test', imputer, balancer)

    dict_new = {
        'X_train': X_train,
        'y_train': y_train,
        'X_valid': X_valid,
        'y_valid': y_valid,
        'X_test': X_test,
        'y_test': y_test
    }

    if ctx['type'] != 'Regression':
        viz_confmatrix = get_confusion_matrix(ctx, model, dict_new, ctx['label'])
        viz_reliability = get_reliability_curve(ctx, model, dict_new)
        viz_roc = get_roc(ctx, model, dict_new)
        roc_reliability = viz_roc + viz_reliability

        hv.save(viz_confmatrix, f'outputs/confmatrix.html')
        hv.save(viz_reliability, f'outputs/reliability.html')
        hv.save(viz_roc, f'outputs/roc.html')
        hv.save(roc_reliability, f'outputs/roc_reliability.html')

    else:
        viz_residuals = get_residuals_plot(ctx, model, dict_new)

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
import glob
import holoviews as hv
import joblib
import json
import mlflow
import pandas as pd

from azureml.core import Run
from azureml.core.run import _OfflineRun
from bokeh.io import export_png

from pathlib import Path
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options


def context(func):
    def get_webdriver():
        try:
            options = Options()
            options.add_argument('--headless')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            options.binary_location = '/opt/google/chrome/chrome'
            return Chrome(options=options,
                executable_path=str(Path("/usr/bin/chromedriver")))
        except:
            return None

    def inner(**kwargs):
        # init mlflow
        if mlflow.active_run() is None:
            mlflow.start_run()
    
        client = mlflow.tracking.MlflowClient()
    
        # get run context
        run = Run.get_context()
        if type(run) != _OfflineRun:
            tags = run.parent.get_tags()
        else:
            tags = {}

        if 'workspace' not in kwargs:
            kwargs['workspace'] = run.experiment.workspace

        if 'primary_metric' not in kwargs:
            kwargs['primary_metric'] = tags['primary_metric'] if 'primary_metric' in tags.keys() else None

        kwargs['chrome'] = get_webdriver()
        kwargs['run'] = run
        kwargs['data'] = client.get_run(mlflow.active_run().info.run_id).data
        kwargs['project'] = tags['project'] if 'project' in tags.keys() else None
        kwargs['type'] = tags['type'] if 'type' in tags.keys() else None
        kwargs['label'] = tags['label'] if 'label' in tags.keys() else None
        
        return func(**kwargs)

    return inner


def data(func):
    def get_df(path):
        df_all = pd.DataFrame()
        deltas = glob.glob(str(path)+"/*")
        for d in deltas:
            print('adding {}'.format(d))
            df_delta = pd.read_csv(d, parse_dates=['runDate'])
            df_all = pd.concat([df_all, df_delta], ignore_index=True)

        df_all['runDate'] = pd.to_datetime(df_all['runDate'])

        return df_all.sort_values('runDate', ascending=False)

    def eval_df(dict_files, fold, imputer, balancer='none'):
        if f'X_{fold}' in dict_files.keys():
            df_X = dict_files[f'X_{fold}']
            df_y = dict_files[f'y_{fold}']
            
            cols = df_X.columns

            df_X = dict_files[f'imputer____{imputer}'].fit_transform(df_X)
            if balancer != 'none':
                df_X, df_y = dict_files[f'balancer____{imputer}_{balancer}'].fit_resample(df_X, df_y)

            df_X = pd.DataFrame(df_X, columns=cols)
            
            return df_X, df_y
        else:
            return None, None

    def inner(**kwargs):
        if kwargs['include_runinfo'] == True:
            df_runinfo = get_df(Path(kwargs['runinfo']))
        
        if kwargs['include_trainlog'] == True:
            df_trainlog = get_df(Path(kwargs['trainlog']))

        runId = df_trainlog.sort_values('runDate', ascending=False).iloc[0]['sweep_get_best_model_runid']
        
        run = Run.get(workspace=kwargs['workspace'], run_id=runId)
        run.download_file('outputs/model.pkl', output_file_path='data/model.pkl')
        run.download_file('outputs/datasets.pkl', output_file_path='data/datasets.pkl')
        run.download_file('outputs/best_run.json', output_file_path='data/best_run.json')

        with open('data/best_run.json', 'r') as f:
            best_run = json.load(f)
            imputer = best_run['imputer']
            balancer = best_run['balancer']

        model = joblib.load('data/model.pkl')
        dict_files = pd.read_pickle('data/datasets.pkl')

        X_train, y_train = eval_df(dict_files, 'train', imputer, balancer)
        X_valid, y_valid = eval_df(dict_files, 'valid', imputer, balancer)
        X_test, y_test = eval_df(dict_files, 'test', imputer, balancer)

        kwargs['data_layer.folds'] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_valid': X_valid,
            'y_valid': y_valid,
            'X_test': X_test,
            'y_test': y_test
        }

        kwargs['data_layer.runinfo'] = df_runinfo
        kwargs['data_layer.trainlog'] = df_trainlog

        return func(**kwargs)

    return inner


def export(**kwargs):
    try:
        if 'plot' in kwargs:
            #hv.save(kwargs['plot'], f'outputs/{kwargs["plot_name"]}.html')
            #export_png(hv.render(kwargs['plot']), filename=f'outputs/{kwargs["plot_name"]}.png', webdriver=kwargs['chrome'])
            return kwargs['plot']
    except RuntimeError:
        pass

    return None


def debug(**kwargs):
    return kwargs
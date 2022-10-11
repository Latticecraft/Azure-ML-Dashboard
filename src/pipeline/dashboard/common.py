import glob
import pandas as pd

from pathlib import Path
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options


class LazyEval:
    def __init__(self, dict_files):
        self.dict_files = dict_files

    def get(self, fold, imputer, balancer='none'):
        df_X = self.dict_files[f'X_{fold}']
        df_y = self.dict_files[f'y_{fold}']
        
        cols = df_X.columns

        df_X = self.dict_files[f'imputer____{imputer}'].fit_transform(df_X)
        if balancer != 'none':
            df_X, df_y = self.dict_files[f'balancer____{imputer}_{balancer}'].fit_resample(df_X, df_y)

        df_X = pd.DataFrame(df_X, columns=cols)
        
        return df_X, df_y


def get_df(path):
    df_all = pd.DataFrame()
    deltas = glob.glob(path+"/*")
    for d in deltas:
        print('adding {}'.format(d))
        df_delta = pd.read_csv((Path(path) / d), parse_dates=['runDate'])
        df_all = pd.concat([df_all, df_delta], ignore_index=True)

    df_all['runDate'] = pd.to_datetime(df_all['runDate'])
    return df_all.sort_values('runDate', ascending=False)


def get_webdriver():
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.binary_location = '/opt/google/chrome/chrome'
    return Chrome(options=options,
        executable_path=str(Path("/usr/bin/chromedriver")))


''' -- Themes not fully supported yet for png exports
def get_theme():
    return Theme(json={
        'attrs' : {
            'Title':{
                'text': f'Generated {datetime.now().strftime("%c")}',
                'text_color': "black",
                'text_font_size': '6pt'
            },
            'Figure' : {
                'background_fill_color': None,
                'background_fill_alpha': 0,
                'border_fill_color': None,
                'border_fill_alpha': 0,
                'outline_line_color': '#444444',
            },
            'Grid': {
                'grid_line_dash': [6, 4],
                'grid_line_alpha': .3,
            },

            'Axis': {
                'major_label_text_color': 'black',
                'axis_label_text_color': 'black',
                'major_tick_line_color': 'black',
                'minor_tick_line_color': 'black',
                'axis_line_color': "black"
            },
            'Legend':{
                'background_fill_color': 'black',
                'background_fill_alpha': 0.5,
                'location' : "center_right",
                'label_text_color': "black"
            }
        }})
'''
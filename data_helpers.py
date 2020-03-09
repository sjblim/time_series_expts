from statsmodels.tsa import stattools
import sklearn
import pandas as pd
import numpy as np

def test_stationarity(df_map, num_samples=10):

    if not isinstance(df_map, dict):
        df_map = {'default': df_map}

    ave_pvals = {}
    for k in df_map:
        df = df_map[k]

        col_samples = np.random.choice(list(df.columns), size=num_samples)
        
        for col in col_samples:
            srs = df[col]

            res = stattools.adfuller(srs)
            ave_pvals[k] = res[1]/num_samples

    return ave_pvals

def detect_outliers(df_map, lookback=252, zscore_thresh=5):

    if not isinstance(df_map, dict):
        df_map = {'default': df_map}

    num_outliers = {}
    for k in df_map:
        df = df_map[k]
        
        ewm = df.ewm(lookback)
        m = ewm.mean()
        std = ewm.std()

        is_outlier = ((np.abs(df-m)/std) >= zscore_thresh)
        num_outliers[k] = is_outlier.sum()

    return num_outliers

def get_ave_autocorr(target, maxlag=10):
    auto_corr = None
    for k in target:
        ac = pd.Series({lag: target[k].autocorr(lag=lag) 
                        for lag in range(1, maxlag+1)})
        
        if auto_corr is None:
            auto_corr = ac / len(target.columns)
        else:
            auto_corr += ac / len(target.columns)
    return auto_corr

def quantile_winsorize(df_map, bounds=None, thresh=0.01):
    
    outputs = {}

    if bounds is None:
        bounds = {}
        
        for k in df_map:
            df = df_map[k]
            lb = df.quantile(q=thresh)
            ub = df.quantile(q=1-thresh)
            bounds[k] = (lb,ub)

    for k in df_map:
        tmp = df_map[k]
        if k in bounds:
            lb,ub = bounds[k]
            tmp = tmp.clip(lower=lb, upper=ub, axis=1)
        outputs[k] = tmp

    return outputs, bounds

def entity_zscore_norm(df_map, norms=None):
    
    outputs = {}

    if norms is None:

        norms = {}

        for k in df_map:
            df = df_map[k]
            m, s = df.mean(), df.std()

            norms[k] = (m, s)

    for k in df_map:

        if k in norms:
            m,s = norms[k]
        else:
            m,s = 0, 1

        outputs[k] = (df_map[k] - m) / s

    return outputs, norms

def combined_zscore_norm(df_map, norms=None):
    
    outputs = {}

    if norms is None:

        norms = {}

        for k in df_map:
            df = df_map[k]
            m, s = df.values.mean(), df.values.std()

            norms[k] = (m, s)

    for k in df_map:

        if k in norms:
            m,s = norms[k]
        else:
            m,s = 0, 1

        outputs[k] = (df_map[k] - m) / s

    return outputs, norms
        
def diff_data(df_map, cols):

    output = {}

    for k in df_map:
        tmp = df_map[k]

        if k in cols:
            
            tmp = tmp.diff(1)
        
        tmp = tmp.iloc[1:, :]
        output[k] = tmp

    return output

def add_lags(df_map, lags=[1]):

    output = {k: df_map[k] for k in df_map}

    for lag in lags:
        for k in df_map:

            lag_name = '{}_{}'.format(k, lag)

            df = df_map[k].shift(lag).fillna(0)
            output[lag_name] = df

    return output

#%%
import os

import matplotlib.pyplot as plt
from statsmodels.tsa import stattools
import sklearn
import pandas as pd
import numpy as np

import configs
import data_helpers as helpers

if __name__ == "__main__":
    
    
    df_map = configs.get_datamap()
    target_col = 'rv5_ss'
    
    # Check data
    print("*** Commencing outlier detection ***")
    print()

    outliers = helpers.detect_outliers(df_map)
    print('Ave Outliers')
    print(outliers)
    
    print()
    print("*** Running stationarity tests ***")
    print()   

    pvals =  helpers.test_stationarity(df_map, num_samples=10)
    print('Pvals:')
    print(pvals)

    print()
    print("*** Visualising autocorrelation ***")
    print()
    target = df_map[target_col]
    auto_corr = helpers.get_ave_autocorr(target, maxlag=20)
    print("Autocorr")
    print(auto_corr)


# %%

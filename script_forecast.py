import pandas as pd

import configs
import data_helpers

from models import PLS, LGBM, MLP, Lasso

if __name__ == "__main__":

    df_map, test_map = configs.get_datamap()

    for m in [df_map, test_map]:
        m['rv5_ss'] = m['rv5_ss'].diff(-1).fillna(0)
    target_col = 'rv5_ss'
    cols2diff = ['open_price']

    # Diff data
    df_map = data_helpers.diff_data(df_map, cols2diff)
    test_map = data_helpers.diff_data(test_map, cols2diff)

    # Winsorize data
    df_map, bounds = data_helpers.quantile_winsorize(df_map)
    del bounds[target_col]  # to avoid winsorizing oos target
    test_map, _ = data_helpers.quantile_winsorize(test_map, bounds)

    # Add zscore normalisation
    df_map, scale = data_helpers.combined_zscore_norm(df_map)
    test_map, _ = data_helpers.combined_zscore_norm(test_map, scale)

    print("*** Commencing PLS forecast ***")
    params = {'n_components': 5}
    
    pls = PLS(params)
    
    r2 = pls.fit(df_map, target_col, valid_fraction=0.1, use_cv=True)

    print("*** Commencing pure ML forecasts ****")
    pls_train = df_map
    pls_test = test_map
    df_map = data_helpers.add_lags(df_map, lags=[1, 2, 3, 4, 5])
    test_map = data_helpers.add_lags(test_map, lags=[1, 2, 3, 4, 5])
    
    params = {'max_depth': -1,
                'min_data_in_leaf': 20,
                'num_leaves': 16, #[16, 32, 64, 128],
                'learning_rate': 0.05,
                'n_estimators': 160,
                'feature_fraction': 0.8,
                'bagging_freq': 0,  # disables
                'bagging_fraction': 1.0,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'min_gain_to_split':0.0,
                'n_jobs': 3,
                'boosting_type': 'gbdt'
            }

    lightgbm1 = LGBM(params)
    r2 = lightgbm1.fit(df_map, target_col, valid_fraction=0.1, 
                        rs_iterations=100)
    
    lasso = Lasso()
    lasso.fit(df_map, target_col, valid_fraction=0.1)


    params = {'hidden_layer_sizes': 20,
              'alpha': 0.001,
              'batch_size': 'auto',
              'learning_rate_init':0.001}
    mlp = MLP(params)
    mlp.fit(df_map, target_col, valid_fraction=0.1, rs_iterations=50)

    print("*** Commencing hybrid forecast ***")
    
    def add_latents(df_map):
        
        latents = pls.get_latents(df_map)
        
        outputs = {k: df_map[k] for k in df_map}
        order = sorted(list(df_map.keys()))
        stocks = df_map[order[0]].columns
        
        for i in range(latents.shape[-1]):
            name = 'latent_{}'.format(i)
            tmp = pd.DataFrame({stock: latents[:, i] for stock in stocks})
            outputs[name] = tmp
        
        return outputs
    
    df_map = add_latents(pls_train)
    test_map = add_latents(pls_test)
    df_map = data_helpers.add_lags(df_map, lags=[1, 2, 3, 4, 5])
    test_map = data_helpers.add_lags(test_map, lags=[1, 2, 3, 4, 5])
    
    # Forecasting models
    params = {'max_depth': -1,
                'min_data_in_leaf': 20,
                'num_leaves': 16, #[16, 32, 64, 128],
                'learning_rate': 0.05,
                'n_estimators': 160,
                'feature_fraction': 0.8,
                'bagging_freq': 0,  # disables
                'bagging_fraction': 1.0,
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
                'min_gain_to_split':0.0,
                'n_jobs': 3,
                'boosting_type': 'gbdt'
            }

    lightgbm1 = LGBM(params)
    r2 = lightgbm1.fit(df_map, target_col, valid_fraction=0.1, 
                        rs_iterations=100)
    
    lasso = Lasso()
    lasso.fit(df_map, target_col, valid_fraction=0.1)


    params = {'hidden_layer_sizes': 20,
              'alpha': 0.001,
              'batch_size': 'auto',
              'learning_rate_init':0.001}
    mlp = MLP(params)
    mlp.fit(df_map, target_col, valid_fraction=0.1, rs_iterations=50)
            
        
            
    

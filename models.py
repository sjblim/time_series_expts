import os
import numpy as np
from sklearn.cross_decomposition import PLSRegression
from lightgbm import LGBMRegressor
from sklearn import linear_model, neural_network
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib
import sklearn.model_selection as select

class PLS:

    def __init__(self, params):
        self.name = "pls"
        self.model = PLSRegression(n_components=params['n_components'])
        self.target_col = None
        
    def _format_data(self, data_map):  
        if self.target_col is None:
            raise ValueError("Target col is None!")
        order = sorted(list(data_map.keys()))
        X = {k:data_map[k] for k in order if k != self.target_col}
        inputs = np.concatenate([X[k] for k in X], axis=1)
        return inputs

    def fit(self, train_map, target_col, valid_fraction=0.2, use_cv=False):
        print("Formatting data")
        self.target_col = target_col

        y = train_map[target_col].values
        X = self._format_data(train_map)

        splitpoint = int(y.shape[0]*(1-valid_fraction))

        y_valid, X_valid = y[splitpoint:], X[splitpoint:]
        y, X = y[:splitpoint], X[:splitpoint]
        
        if use_cv:
            print('Running grid search')
            param_dist = self.get_hyperparam_ranges()
            self.model = select.GridSearchCV(self.model,
                                     param_grid=param_dist,
                                     cv=3,
                                     n_jobs=2)
        
        print("Fitting PLS")
        self.model.fit(X, y)

        if valid_fraction != 0:
            print("Scoring on validation data")
            r2 = self.model.score(X_valid, y_valid)

            print("R2 for PLS:", r2)
            return r2
        else:
            print("No validation data")
            return 0.0

    def predict(self, data_map):
        X = self._format_data(data_map)
        return self.model.predict(X)

    def get_latents(self, data_map):
        X = self._format_data(data_map)
        return self.model.transform(X)

    def get_save_name(self, model_folder):
        return os.path.join(model_folder, self.name+".joblib")

    def save(self, model_folder):
        name = self.get_save_name(model_folder)
        joblib.dump(self.model, name)

    def load(self, model_folder):
        name = self.get_save_name(model_folder)
        self.model = joblib.load(name)
    
    @classmethod
    def get_hyperparam_ranges(cls):
        params = {'n_components': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 40]}
        return params

class LGBM:

    def __init__(self, params):
        self.name = "lgbm"
        
        # Fix this
        learning_rate = params['learning_rate']
        n_estimators = params['n_estimators']
        min_data_in_leaf = params['num_leaves']
        # key params
        num_leaves = params['num_leaves']
        min_gain_to_split = params['min_gain_to_split']
        max_depth = params['max_depth']

        # speed vs accuracy tradeoffs
        bagging_freq = params['bagging_freq']
        bagging_frac = params['bagging_fraction']
        feature_frac = params['feature_fraction']

        # Regularisation
        reg_alpha = params['reg_alpha']
        reg_lambda = params['reg_lambda']
        n_jobs = params['n_jobs']#3  # -1
        boosting_type = params['boosting_type'] #'gbdt'  #["dart", 'gbdt', 'goss', 'rf']

        self.model = LGBMRegressor(learning_rate=learning_rate,
                                    n_estimators=n_estimators, 
                                    num_leaves=num_leaves,
                                    min_data_in_leaf=min_data_in_leaf,
                                    max_depth=max_depth,
                                    min_split_gain=min_gain_to_split,
                                    bagging_fraction=bagging_frac,
                                    bagging_freq=bagging_freq,
                                    feature_frac=feature_frac,
                                    reg_alpha=reg_alpha,
                                    reg_lambda=reg_lambda,
                                    n_jobs=n_jobs,
                                    boosting_type=boosting_type
                                    )
        self.target_col = None

    def _split_data_maps(self, data_map, split_fraction):
        train = {}
        test = {}
        
        order = sorted(list(k for k in data_map.keys()))

        length = len(data_map[order[0]])
        splitpoint = int(length*split_fraction)
        for k in order:
            train[k] = data_map[k].iloc[:splitpoint]
            test[k] = data_map[k].iloc[splitpoint:]

        return train, test

    def _format_data(self, data_map):  
        if self.target_col is None:
            raise ValueError("Target col is None!")
        
        order = sorted(list(k for k in data_map.keys() if k != self.target_col))

        inputs = []
        num_stocks = data_map[order[0]].shape[-1]
        for i in range(num_stocks):
            stock_data = []
            for k in order:
                arr = data_map[k].iloc[:, i]
                stock_data.append(arr.values.reshape(-1, 1))
            inputs.append(np.concatenate(stock_data, axis=1))

        inputs = np.concatenate(inputs, axis=0)

        return inputs
    
    def _format_target(self, data_map):
        if self.target_col is None:
            raise ValueError("Target col is None!")

        target = data_map[self.target_col].values

        return target.reshape(-1, 1) # stacked targets

    def fit(self, data_map, target_col, valid_fraction=0.2, rs_iterations=-1):
        print("Formatting data")
        self.target_col = target_col

        print("Splitting data map")
        train_map, valid_map = self._split_data_maps(data_map, valid_fraction)

        y = self._format_target(train_map)
        X = self._format_data(train_map)

        print("Fitting", self.name)
        
        if rs_iterations > 0:
            
            param_dist = self.get_hyperparam_ranges()
            param_combinations = np.prod([len(param_dist[k]) for k in param_dist])
            rs_iterations = min(param_combinations, rs_iterations)
            
            print("Running {} iterations of random search".format(
                    rs_iterations))

            self.model = select.RandomizedSearchCV(self.model,
                                     param_distributions=param_dist,
                                     n_iter=rs_iterations,
                                     cv=3,
                                     n_jobs=2)
        self.model.fit(X, y)

        if valid_fraction != 0:
            y_valid = self._format_target(valid_map)
            X_valid = self._format_data(valid_map)

            print("Scoring on validation data")
            r2 = self.model.score(X_valid, y_valid)

            print("R2 for {}:".format(self.name.upper()), r2)
            return r2
        else:
            print("No validation data")
            return 0.0

    def predict(self, data_map):
        X = self._format_data(data_map)
        return self.model.predict(X)

    def get_save_name(self, model_folder):
        return os.path.join(model_folder, self.name+".joblib")

    def save(self, model_folder):
        name = self.get_save_name(model_folder)
        joblib.dump(self.model, name)

    def load(self, model_folder):
        name = self.get_save_name(model_folder)
        self.model = joblib.load(name)
    
    @classmethod
    def get_hyperparam_ranges(cls):
        param_grid = {  'max_depth': [-1],
                        'min_data_in_leaf': [20, 40, 80],
                        'num_leaves': [8, 16, 32, 64, 128],
                        'learning_rate': [1.0, 0.1, 0.05, 0.01],
                        'n_estimators': [50, 100, 200],
                        'feature_fraction': [0.2, 0.4, 0.6, 0.8],
                        'bagging_freq': [0],  # disables
                        'bagging_fraction': [1.0], # disables
                        'reg_alpha': [0.0, 1.0, 0.1,0.01],
                        'reg_lambda': [0.0,1.0, 0.1,0.01],
                        'min_gain_to_split':[0.001],
                        'n_jobs':[2],
                        'boosting_type':['gbdt', 'dart']
                    }

        return param_grid

class Lasso(LGBM):
    def __init__(self):
        self.name = "lasso"

        self.model = linear_model.LassoCV(n_jobs=3, cv=3, max_iter=10000)    
    @classmethod
    def get_hyperparam_ranges(cls):
        return {}

class MLP(LGBM):

    def __init__(self, params):
        self.name = "mlp"
        hidden_layer_sizes = params['hidden_layer_sizes']
        alpha = params['alpha']
        batch_size = params['batch_size']
        learning_rate_init = params['learning_rate_init']
        self.model = neural_network.MLPRegressor(
            activation="relu", 
            solver="adam",
            max_iter=1000, 
            early_stopping=True, 
            tol=1e-8,
            hidden_layer_sizes=hidden_layer_sizes,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init)  

        self.target_col = None

    @classmethod
    def get_hyperparam_ranges(cls):
        param_dist = {'hidden_layer_sizes': [5, 10, 25, 50, 100],
                      'alpha': np.logspace(-6, 1, 8),
                      'batch_size': [128, 256, 512, 1024, 2048],  # minibatch sizes
                      'learning_rate_init': np.logspace(-5, 1, 7)}

        return param_dist
        
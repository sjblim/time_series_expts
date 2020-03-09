"""
sklearn_regression.py


Created by limsi on 25/10/2018
"""

import os
import pandas as pd
import numpy as np
from time import time
import argparse

# ML metrics
from sklearn import metrics
import sklearn.model_selection as select
from sklearn import linear_model, svm, neural_network, ensemble, gaussian_process
from sklearn.multioutput import MultiOutputRegressor
from sklearn.externals import joblib


def get_cross_validation_estimates(model_name, max_iterations, n_jobs=1):
    epsilon = 10 ** -3

    if model_name == "ridge":

        model = linear_model.Ridge()
        param_dist = {'alpha': [0.0, 0.001, 0.01, 0.1, 1.0, 10]}

    elif model_name == 'lasso':
        model = linear_model.Lasso()
        param_dist = {'alpha': [0.0, 0.001, 0.01, 0.1, 1.0, 10]}

    elif model_name == "svr":

        model = svm.SVR()
        param_dist = {'kernel': ["linear", "poly", "rbf", "sigmoid"],
                      'degree': [3, 5, 7],
                      'gamma': np.logspace(-3, 1, 5),
                      'C': np.logspace(-3, 1, 5),
                      'epsilon': np.logspace(-3, 1, 5)}

    elif model_name == "sgdregressor":  # linear SVM for large datasets

        model = linear_model.SGDRegressor(max_iter=100,
                                          penalty='l1',  # L1 loss SGD regressor
                                          early_stopping=True,
                                          tol=epsilon,
                                          learning_rate='invscaling')  # number of epochs
        param_dist = {'alpha': np.logspace(-6, 0, 7),
                      'eta0': np.logspace(1, -3, 5),
                      'power_t': [0.25, 0.5, 1, 2.0]}

        param_dist = {"estimator__" + k: param_dist[k] for k in param_dist}
        model = MultiOutputRegressor(model)

    elif model_name == "rvm":
        model = linear_model.ARDRegression()
        param_dist = {'alpha_1': [1e-9, 1e-6, 1e-3],
                      'alpha_2': [1e-9, 1e-6, 1e-3],
                      'lambda_1': [1e-9, 1e-6, 1e-3],
                      'lambda_2': [1e-9, 1e-6, 1e-3]}

    elif model_name == "mlp":
        model = neural_network.MLPRegressor(activation="relu", solver="adam",
                                            max_iter=200, early_stopping=True, tol=epsilon)  #

        param_dist = {'hidden_layer_sizes': [5, 10, 25, 50, 100, 150],
                      'alpha': np.logspace(-6, 1, 8),
                      'batch_size': [128, 256, 512, 1024, 2048],  # minibatch sizes
                      'learning_rate_init': np.logspace(-5, 1, 7)}
    elif model_name == "gp":

        return gaussian_process.GaussianProcessRegressor()  # hyperparams fitted internally...

    elif model_name == "rf":
        model = ensemble.RandomForestRegressor()
        # From https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
        # Number of trees in random forest
        n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
        # Number of features to consider at every split
        max_features = ['auto', 'sqrt']
        # Maximum number of levels in tree
        max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
        max_depth.append(None)
        # Minimum number of samples required to split a node
        min_samples_split = [2, 5, 10]
        # Minimum number of samples required at each leaf node
        min_samples_leaf = [1, 2, 4]
        # Method of selecting samples for training each tree
        bootstrap = [True, False]
        # Create the random grid
        param_dist = {'n_estimators': n_estimators,
                      'max_features': max_features,
                      'max_depth': max_depth,
                      'min_samples_split': min_samples_split,
                      'min_samples_leaf': min_samples_leaf,
                      'bootstrap': bootstrap}

    param_combinations = np.prod([len(param_dist[k]) for k in param_dist])
    max_iterations = min(param_combinations, max_iterations)
    print(max_iterations)

    return select.RandomizedSearchCV(model,
                                     param_distributions=param_dist,
                                     n_iter=max_iterations,
                                     n_jobs=n_jobs,  # use just 2 processors
                                     verbose=3)


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def fit(y_scaled, x_scaled, method, random_search_iterations, n_jobs=1):

    random_search = get_cross_validation_estimates(method, random_search_iterations,
                                                   n_jobs=n_jobs)

    start = time()
    random_search.fit(x_scaled, y_scaled)

    if method != "gp":
        print("{}: took {} seconds for {} candidates"
              " parameter settings.".format(method, (time() - start), random_search_iterations))
        report(random_search.cv_results_)
    else:
        print("GP Params: {}".format(random_search.get_params()))

    return random_search


def evaluate(y_scaled, x_scaled, random_search):
    y_pred = random_search.predict(x_scaled)

    mse = metrics.mean_squared_error(y_scaled, y_pred)
    print("Normalised Test MSE = {}".format(mse))

    return mse


def run_calibration(train_data, test_data, method, random_search_iterations, model_path):

    x_scaled, y_scaled = train_data
    x_test, y_test = test_data

    random_search = fit(y_scaled, x_scaled, method, random_search_iterations)

    mse = evaluate(y_test, x_test, random_search)

    joblib.dump(random_search, model_path)

    return mse


def evaluate_trained_model(test_data, model_path):
    X, y = test_data

    model = joblib.load(model_path)

    mse = evaluate(y, X, model)

    return mse


def predict_trained_model(X, model_path):

    model = joblib.load(model_path)

    predictions = model.predict(X)

    return predictions

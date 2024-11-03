"""
PARKINSONS TELEMONITORING MACHINE LEARNING PROJECT
https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring
Davide Paltrinieri 165005
"""

from ucimlrepo import fetch_ucirepo

import pandas as pd
import numpy as np
import logging
import os
import argparse

import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR


def show_graphs(x, y, model_name, target_name):
    graphs_dir_path = "graphs"
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    plt.figure()
    plt.scatter(x, y, marker='o', linestyle='-', color='b', label=model_name, s=5)
    plt.title(model_name + " " + target_name)
    plt.xlabel('y predicted')
    plt.ylabel('y true')
    plt.grid(True)
    plt.legend()
    if not os.path.exists(graphs_dir_path):
        os.makedirs(graphs_dir_path)
    plt.savefig(graphs_dir_path + "/" + target_name + "_" + model_name + '.png')

def prepocessing_phase(file_data_path, fetaures_removed):
    try:
        with open(file_data_path, 'r') as f:
            lines = f.readlines()
            column_names = lines[0].strip().split(',')
            """
            column_names = [
            "subject#", "age", "sex", "test_time", "motor_UPDRS", "total_UPDRS", 
            "Jitter(%)", "Jitter(Abs)", "Jitter:RAP", "Jitter:PPQ5", "Jitter:DDP", 
            "Shimmer", "Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "Shimmer:APQ11", 
            "Shimmer:DDA", "NHR", "HNR", "RPDE", "DFA", "PPE"
            ]
            """
            logging.debug("Dataset " + file_data_path + " opened correctly.")
    except FileNotFoundError as e:
        logging.debug("Dataset " + file_data_path + " not found.") 
        raise e
    
    df = pd.read_csv(file_data_path, names=column_names, header=0)
    
    df['sex'] = df['sex'].astype('category')
    df = df.drop(columns=fetaures_removed)

    X = df.drop(columns=['motor_UPDRS', 'total_UPDRS'])
        
    y_motor = df['motor_UPDRS']
    y_total = df['total_UPDRS']
    
    X_train, X_test, y_motor_train, y_motor_test, y_total_train, y_total_test = train_test_split(
        X, y_motor, y_total, test_size=0.3, random_state=42
    ) 
    
    X_train, X_test = standardization(X_train, X_test)
    return X_train, y_motor_train, y_total_train, X_test, y_motor_test, y_total_test, X.columns

def standardization(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def calculate_metrics(y_true, y_pred):
    calculated_metrics={
        "neg_mae": -metrics.mean_absolute_error(y_true, y_pred),
        "neg_mse": -metrics.mean_squared_error(y_true, y_pred),
        "neg_rmse": -np.sqrt(metrics.mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred)
    }
    return calculated_metrics

def define_best_model(pipe_dic, X_train, y_train):
    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_root_mean_squared_error', 'r2']
    results = {}
    print("\nCross Validation")
    for model_name, pipe in pipe_dic.items(): # {model_name: [pipeline_fs, pipeline]}
        for i in range(2):
            scores = cross_validate(pipe[i], X_train, y_train, cv=5, scoring=scoring, return_train_score=True, n_jobs=-1)
            name = model_name
            if i == 0:
                name = model_name + "_fs"
            print(name)
            results[name] = scores
            
    for model_name, scores in results.items():
        logging.debug(f"Model --> " + model_name)  
        for metric in scoring:
            mean_score = np.mean(scores[f'test_{metric}'])
            logging.debug(f"{metric}: {mean_score:.4f}")
        logging.debug("\n")
    
    best_score = float('-inf')
    best_pipe = None
    for model_name, scores in results.items():
        # come metrica di confronto scelgo r2
        mean_r2 = np.mean(scores['test_r2'])
        if mean_r2 > best_score:
            best_score = mean_r2
            best_pipe = model_name
    
    print("\nbest pipe: " + best_pipe)
    if best_pipe.endswith("_fs"):
        return model_name.rstrip('_fs'), 0
    else:
        return model_name, 1

def create_sfs(model, cv, scoring='r2', n_features='auto'):
    return SequentialFeatureSelector(model, scoring=scoring, n_features_to_select=n_features, cv=cv, n_jobs=-1)

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='pt_dataset\parkinsons_updrs.data')
    return parser

def define_selected_feature(grid_search, column_names):
    feature_selector = grid_search.named_steps['feature_selection']
    selected_features_indices = feature_selector.get_support(indices=True)
    return [column_names[i] for i in selected_features_indices]
    
def final_evalutaion(pipe, X_test, y_test, y_name, grid_search, fs, column_names):
    y_predicted = pipe.predict(X_test)
    scores = calculate_metrics(y_test, y_predicted)
    pipe_name = pipe.named_steps['model'].__class__.__name__
    if not fs:
        pipe_name = pipe_name + "_FeatureSelection"
    
    show_graphs(y_test, y_predicted, pipe_name, y_name)
    
    logging.debug(f"The best model for {y_name} is {pipe.named_steps['model'].__class__.__name__}")
    logging.debug("The hyperparameters are: " + str(grid_search.get_params().get("model")))
    if fs:
        logging.debug("The model does not use the feature selection")
    else:
        logging.debug("The model uses the feature selection and the features selected are: " + str(define_selected_feature(grid_search, column_names)))
    
    logging.debug("The final evaluation on testing set:")
    for metric_name, metric in scores.items():
        logging.debug(f"\n{metric_name} --> {metric}")

def save_models(models):
    folder = "models"
    if not os.path.exists(folder):
        os.makedirs(folder)
    for model_name, pipe in models.items():
        joblib.dump(pipe, folder + "/" + model_name + ".pkl")

def main():
    fetaures_removed = ["subject#"]
    y_names = ["motor_UPDRS", "total_UPDRS"]
    
    args = setup_parser().parse_args()
    
    try:
        X_train, y_motor_train, y_total_train, X_test, y_motor_test, y_total_test, column_names = prepocessing_phase(args.dataset, fetaures_removed)
    except FileNotFoundError as e:
        logging.debug(e)
        return
    
    y_train = [y_motor_train, y_total_train]
    y_test = [y_motor_test, y_total_test]
    
    models = {
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}
        },
        'KNeighborsRegressor': {
            'model': KNeighborsRegressor(),
            'params': {
                'model__n_neighbors': [3, 5, 7, 9]
            }
        },
        'SVR': {
            'model': SVR(),
            'params': {
                'model__C': [1, 10],
                'model__gamma': [0.001, 0.01]
            }
        },
        'DecisionTreeRegressor': {
            'model': DecisionTreeRegressor(),
            'params': {
                'model__max_depth': [10, 20, None],
                'model__min_samples_split': [2, 5]
            }
        },
        'RandomForestRegressor': {
            'model': RandomForestRegressor(),
            'params': {
                'model__n_estimators': [50, 100, 200],
                'model__min_samples_split': [2, 5, 10]
            }
        }
    }
    
    cv = KFold(n_splits=3, shuffle=True, random_state=42)
    final_models = {}
    
    for i in range(2):
        logging.debug("\n" + "#" * 80  + " " + y_names[i] + " " + "#" * 80)
        pipelines_dic = {}
        grid_results = {}
        for model_name, model_info in models.items():
            print("\nY target: " + y_names[i] + " Model: " + model_name)
            X_train_copy = X_train.copy()
            y_train_copy = y_train[i].copy()
            pipe = []
            
            sfs = create_sfs(model_info['model'], cv)
            
            pipeline_fs = Pipeline([
                ('feature_selection', sfs),
                ('model', model_info['model'])
            ])
            pipe.append(pipeline_fs)
            grid_search_fs = GridSearchCV(pipeline_fs, param_grid=model_info['params'], cv=cv, scoring='r2', refit=True, n_jobs=-1)
            grid_search_fs.fit(X_train_copy, y_train_copy)

            pipeline = Pipeline([
                ('model', model_info['model'])
            ])
            pipe.append(pipeline)
            grid_search = GridSearchCV(pipeline, param_grid=model_info['params'], cv=cv, scoring='r2', refit=True, n_jobs=-1)
            grid_search.fit(X_train, y_train[i])
            
            grid_results[model_name] = [grid_search_fs.best_estimator_, grid_search.best_estimator_]
            pipelines_dic[model_name] = pipe # {model_name: [pipeline_fs, pipeline]}
        
        best_pipe, fs = define_best_model(pipelines_dic, X_train, y_train[i])                
        pipelines_dic[best_pipe][fs].fit(X_train, y_train[i])
        final_models[y_names[i]] = pipelines_dic[best_pipe][fs]
        final_evalutaion(pipelines_dic[best_pipe][fs], X_test, y_test[i], y_names[i], grid_results[best_pipe][fs], fs, column_names)
    
    save_models(final_models)
        
if __name__ == '__main__':
    logging.basicConfig(filename='data.log', filemode='w', level=logging.DEBUG, format='%(message)s')
    main()
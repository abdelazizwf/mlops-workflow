import mlflow
import numpy as np
import optuna
import pandas as pd
from dvc.api import params_show
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

mlflow.set_tracking_uri("http://localhost:8080")

train_data = pd.read_csv("prepared_data/train.csv")
val_data = pd.read_csv("prepared_data/val.csv")

X_train = train_data.drop(columns=["Survived", "PassengerId"])
y_train = train_data["Survived"]

X_val = val_data.drop(columns=["Survived", "PassengerId"])
y_val = val_data["Survived"]

X = pd.concat([X_train, X_val], axis=0, ignore_index=True)
y = pd.concat([y_train, y_val], axis=0, ignore_index=True)

def objective(trial):
    hparams = {
        "kernel": trial.suggest_categorical("kernel", ["linear", "rbf", "poly"]),
        "C": trial.suggest_float("C", 1e-2, 1e2, log=True),
    }
    
    kernel = hparams["kernel"]
    if kernel != "linear":
        hparams["gamma"] = trial.suggest_float("gamma", 1e-5, 1, log=True)
    if kernel == "poly":
        hparams["degree"] = trial.suggest_int("degree", 1, 10)
    
    with mlflow.start_run(nested=True):
        mlflow.log_params(hparams)
        
        model = SVC(**hparams, max_iter=2000)
        score = np.mean(cross_val_score(model, X, y))
        
        mlflow.log_metric("cross_val_accuracy", score)
    
    return score

with mlflow.start_run():
    study = optuna.create_study(direction="maximize")
    params = params_show(stages="tune_hparams")["training"]
    n_trials = params["n_trials"]
    study.optimize(objective, n_trials=n_trials)
    best_params = study.best_params
    best_score = study.best_value
    mlflow.log_params(best_params)
    mlflow.log_metric("cross_val_accuracy", best_score)

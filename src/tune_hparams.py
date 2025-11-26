import mlflow
import numpy as np
import optuna
import pandas as pd
from omegaconf import OmegaConf
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC

conf  = OmegaConf.load("./params.yaml")
tracking_uri = conf.tracking_server.uri
experiment_name = conf.tracking_server.experiment_name
n_trials = conf.training.n_trials

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

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
        "max_iter": trial.suggest_int("max_iter", 1000, 5000),
    }
    
    kernel = hparams["kernel"]
    if kernel != "linear":
        hparams["gamma"] = trial.suggest_float("gamma", 1e-5, 1, log=True)
    if kernel == "poly":
        hparams["degree"] = trial.suggest_int("degree", 1, 10)
    
    with mlflow.start_run(nested=True) as run:
        mlflow.log_params(hparams)
        
        model = SVC(**hparams)
        score = np.mean(cross_val_score(model, X, y))
        
        mlflow.log_metric("cross_val_accuracy", score)
        
        trial.set_user_attr("run_name", run.info.run_name)
    
    return score

with mlflow.start_run():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    best_trial = study.best_trial
    conf.model_params.svc = best_trial.params
    OmegaConf.save(conf, "./params.yaml")
    
    best_trial.params.update(best_trial.user_attrs)
    mlflow.log_params(best_trial.params)
    mlflow.log_metric("cross_val_accuracy", best_trial.value)

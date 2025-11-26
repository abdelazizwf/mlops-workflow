import mlflow
import pandas as pd
from omegaconf import OmegaConf
from sklearn.svm import SVC

conf = OmegaConf.load("./params.yaml")
tracking_uri = conf.tracking_server.uri
experiment_name = conf.tracking_server.experiment_name
svc_params = conf.model_params.svc

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

train_data = pd.read_csv("prepared_data/train.csv")
val_data = pd.read_csv("prepared_data/val.csv")

X_train = train_data.drop(columns=["Survived", "PassengerId"])
y_train = train_data["Survived"]

X_val = val_data.drop(columns=["Survived", "PassengerId"])
y_val = val_data["Survived"]

with mlflow.start_run() as run:
    mlflow.log_params(svc_params)
    
    model = SVC(**svc_params)
    model.fit(X_train, y_train)
    train_accuracy = model.score(X_train, y_train)
    val_accuracy = model.score(X_val, y_val)
    
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    })

print(f"Train Accuracy = {train_accuracy:.3f}\nValidation Accuracy = {val_accuracy:.3f}")

import mlflow
import pandas as pd
from dvc.api import params_show
from sklearn.svm import SVC

mlflow.set_tracking_uri("http://localhost:8080")

train_data = pd.read_csv("prepared_data/train.csv")
val_data = pd.read_csv("prepared_data/val.csv")

X_train = train_data.drop(columns=["Survived", "PassengerId"])
y_train = train_data["Survived"]

X_val = val_data.drop(columns=["Survived", "PassengerId"])
y_val = val_data["Survived"]

hparams = params_show()["model"]["svc"]
model = SVC(**hparams, max_iter=2000)

with mlflow.start_run() as run:
    model.fit(X_train, y_train)
    
    train_accuracy = model.score(X_train, y_train)
    val_accuracy = model.score(X_val, y_val)
    
    mlflow.log_metrics({
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
    })

print(f"Train Accuracy = {train_accuracy:.3f}\nValidation Accuracy = {val_accuracy:.3f}")

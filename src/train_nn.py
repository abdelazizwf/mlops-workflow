import mlflow
import pandas as pd
import torch
from dvc.api import params_show
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader, TensorDataset

from model import Model

mlflow.set_tracking_uri("http://localhost:8080")

train_data = pd.read_csv("prepared_data/train.csv")
val_data = pd.read_csv("prepared_data/val.csv")

X_train = torch.tensor(
    train_data.drop(columns=["Survived", "PassengerId"]).to_numpy(),
    dtype=torch.float32,
)
y_train = torch.tensor(train_data["Survived"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
train_dataset = TensorDataset(X_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

X_val = torch.tensor(
    val_data.drop(columns=["Survived", "PassengerId"]).to_numpy(),
    dtype=torch.float32
)
y_val = torch.tensor(val_data["Survived"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
val_dataset = TensorDataset(X_val, y_val)
val_dataloader = DataLoader(val_dataset, batch_size=64)

hparams = params_show()["model"]["nn"]
model = Model(**hparams)

with mlflow.start_run() as run:
    logger = MLFlowLogger(
        run_id=run.info.run_id, tracking_uri=mlflow.get_tracking_uri())
    trainer = Trainer(
        max_epochs=10, logger=logger,
        val_check_interval=None, num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)
    trainer.test(model, dataloaders=val_dataloader)

import mlflow
import torch
from dvc.api import params_show
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader
from torchinfo import summary

from model import Model

mlflow.set_tracking_uri("http://localhost:8080")

train_dataset = torch.load("tensor_data/train_dataset.pt", weights_only=False)
val_dataset = torch.load("tensor_data/val_dataset.pt", weights_only=False)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64)

hparams = params_show()["model"]["nn"]
model = Model(**hparams)

with mlflow.start_run() as run:
    logger = MLFlowLogger(
        run_id=run.info.run_id, tracking_uri=mlflow.get_tracking_uri()
    )
    mlflow.log_text(
        str(summary(model.model, (64, 36), verbose=0)),
        "model_summary.txt",
    )
    trainer = Trainer(
        max_epochs=10, logger=logger,
        val_check_interval=None, num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)
    trainer.test(model, dataloaders=val_dataloader)

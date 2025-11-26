import mlflow
import torch
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchinfo import summary

from model import Model

conf = OmegaConf.load("./params.yaml")
tracking_uri = conf.tracking_server.uri
experiment_name = conf.tracking_server.experiment_name
nn_params = conf.model_params.nn
batch_size = conf.training.batch_size

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

train_dataset = torch.load("tensor_data/train_dataset.pt", weights_only=False)
val_dataset = torch.load("tensor_data/val_dataset.pt", weights_only=False)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

model = Model(**nn_params)

with mlflow.start_run() as run:
    logger = MLFlowLogger(
        run_id=run.info.run_id, tracking_uri=tracking_uri
    )
    mlflow.log_text(
        str(summary(model.model, (batch_size, 36), verbose=0)),
        "model_summary.txt",
    )
    trainer = Trainer(
        max_epochs=10, logger=logger,
        val_check_interval=None, num_sanity_val_steps=0,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=train_dataloader)
    trainer.test(model, dataloaders=val_dataloader)

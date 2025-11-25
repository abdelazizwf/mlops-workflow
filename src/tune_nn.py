from copy import deepcopy

import mlflow
import optuna
import torch
from dvc.api import params_show
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from torch.utils.data import DataLoader

from model import Model

mlflow.set_tracking_uri("http://localhost:8080")

train_dataset = torch.load("tensor_data/train_dataset.pt", weights_only=False)
val_dataset = torch.load("tensor_data/val_dataset.pt", weights_only=False)

train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=64)

def objective(trial):
    hparams = {
        "lr": trial.suggest_float("lr", 5e-3, 5e-1, log=True),
        "momentum": trial.suggest_float("momentum", 0.1, 1.0),
    }
    with mlflow.start_run(nested=True) as run:
        logger = MLFlowLogger(run_id=run.info.run_id, tracking_uri=mlflow.get_tracking_uri())
        model = Model(**hparams)
        trainer = Trainer(
            max_epochs=10, logger=logger,
            val_check_interval=None, num_sanity_val_steps=0,
            enable_checkpointing=False, log_every_n_steps=10,
        )
        trainer.fit(model, train_dataloaders=train_dataloader)
        trainer.test(model, dataloaders=val_dataloader)
        test_loss = logger.experiment.get_metric_history(run.info.run_id, "test_loss")[-1].value
        trial.set_user_attr("run_id", run.info.run_id)
        trial.set_user_attr("model_weights", deepcopy(model.model.state_dict()))
    return test_loss

with mlflow.start_run():
    study = optuna.create_study()
    params = params_show(stages="tune_hparams")["training"]
    n_trials = params["n_trials"]
    study.optimize(objective, n_trials=n_trials)
    
    best_params = study.best_params
    best_score = study.best_value
    user_attrs = study.best_trial.user_attrs
    
    best_weights = user_attrs["model_weights"]
    model = Model(**best_params)
    model.model.load_state_dict(best_weights)
    model_in = next(iter(train_dataloader))[0]
    model_out = model.predict_step(model_in)
    signture = mlflow.models.infer_signature(
        model_in.detach().numpy(),
        model_out.detach().numpy()
    )
    
    best_params["best_run_url"] = mlflow.get_tracking_uri() + "/#/experiments/" + str(0) + "/runs/" + user_attrs["run_id"]
    mlflow.log_params(best_params)
    mlflow.log_metric("test_loss", best_score)
    mlflow.pytorch.log_model(
        model,
        name="basic_tuned_nn",
        registered_model_name="nn-1",
        signature=signture
    )

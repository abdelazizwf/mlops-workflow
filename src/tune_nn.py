from copy import deepcopy

import mlflow
import optuna
import torch
from lightning import Trainer
from lightning.pytorch.loggers import MLFlowLogger
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from model import Model

conf  = OmegaConf.load("./params.yaml")
tracking_uri = conf.tracking_server.uri
experiment_name = conf.tracking_server.experiment_name
n_trials = conf.training.n_trials
batch_size = conf.training.batch_size
epochs = conf.training.epochs

mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment(experiment_name)

train_dataset = torch.load("tensor_data/train_dataset.pt", weights_only=False)
val_dataset = torch.load("tensor_data/val_dataset.pt", weights_only=False)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

def objective(trial):
    hparams = {
        "lr": trial.suggest_float("lr", 5e-3, 5e-1, log=True),
        "momentum": trial.suggest_float("momentum", 0.1, 1.0),
    }
    with mlflow.start_run(nested=True) as run:
        logger = MLFlowLogger(run_id=run.info.run_id, tracking_uri=tracking_uri)
        model = Model(**hparams)
        trainer = Trainer(
            max_epochs=epochs, logger=logger,
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
    study.optimize(objective, n_trials=n_trials)
    
    best_trial = study.best_trial
    conf.model_params.nn = best_trial.params
    OmegaConf.save(conf, "./params.yaml")
    
    best_weights = best_trial.user_attrs["model_weights"]
    model = Model(**best_trial.params)
    model.model.load_state_dict(best_weights)
    model_in = next(iter(train_dataloader))[0]
    model_out = model.predict_step(model_in)
    signture = mlflow.models.infer_signature(
        model_in.detach().numpy(),
        model_out.detach().numpy()
    )
    
    best_trial.params["best_run_url"] = (
        mlflow.get_tracking_uri() + "/#/experiments/" + str(0) + "/runs/" + best_trial.user_attrs["run_id"]
    )
    mlflow.log_params(best_trial.params)
    mlflow.log_metric("test_loss", best_trial.value)
    mlflow.pytorch.log_model(
        model,
        name="basic_tuned_nn",
        code_paths=["./src/model.py"],
        registered_model_name="nn-1",
        signature=signture
    )

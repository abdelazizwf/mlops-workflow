from lightning import LightningModule
from torch import nn, optim
from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    BinaryROC,
)


class Model(LightningModule):
    
    def __init__(self, lr, momentum):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(36, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        self.criterion = nn.BCEWithLogitsLoss()
        
        self.save_hyperparameters()
        
        self.test_metircs = MetricCollection(
            {
                "accuracy": BinaryAccuracy(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "f1": BinaryF1Score(),
                "auroc": BinaryAUROC(),
            },
            prefix="test_",
        )
        self.train_metrics = self.test_metircs.clone(prefix="train_")
        
        self.confusion_matrix = BinaryConfusionMatrix()
        self.pr_curve = BinaryPrecisionRecallCurve()
        self.roc = BinaryROC()
    
    def forward(self, X):
        return self.model(X)
    
    def configure_optimizers(self):
        return optim.SGD(
            self.parameters(), lr=self.hparams["lr"], momentum=self.hparams["momentum"]
        )
    
    def training_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        loss = self.criterion(pred, y)
        self.log("train_loss", loss.item(), prog_bar=True)
        self.train_metrics.update(pred, y)
        self.log_dict(self.train_metrics, on_step=False, on_epoch=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        X, y = batch
        pred = self(X)
        loss = self.criterion(pred, y)
        self.log("test_loss", loss.item())
        self.test_metircs.update(pred, y)
        self.log_dict(self.test_metircs, on_step=False, on_epoch=True)
        for metric in [
            self.confusion_matrix, self.pr_curve, self.roc,
        ]:
            metric.update(pred, y.int())
        
    def on_test_epoch_end(self):
        for metric in [
            self.confusion_matrix, self.pr_curve, self.roc,
        ]:
            fig, _ = metric.plot()
            path = f"plots/{metric.__class__.__name__}.png"
            self.logger.experiment.log_figure(
                figure=fig, artifact_file=path, run_id=self.logger.run_id,
            )

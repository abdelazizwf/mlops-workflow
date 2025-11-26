import mlflow
import pandas as pd
import torch
from omegaconf import OmegaConf

conf  = OmegaConf.load("./params.yaml")
tracking_uri = conf.tracking_server.uri
model_name = conf.predict.model_name
model_version = conf.predict.model_version

mlflow.set_tracking_uri(tracking_uri)

test_data = pd.read_csv("prepared_data/test.csv", index_col="PassengerId")
X_test = torch.tensor(
    test_data.to_numpy(),
    dtype=torch.float32,
)
test_dataset = torch.utils.data.TensorDataset(X_test)

model = mlflow.pytorch.load_model(
    f"models:/{model_name}/{model_version}"
)

preds = torch.nn.functional.sigmoid(
    model.predict_step(test_dataset[:][0])
).squeeze().detach().round().type(torch.int32).tolist()

pd.DataFrame(
    {"PassengerId": test_data.index, "Survived": preds},
).to_csv("./subs.csv", index=False)

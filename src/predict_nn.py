import mlflow
import pandas as pd
import torch
from dvc.api import params_show

mlflow.set_tracking_uri("http://localhost:8080")

test_data = pd.read_csv("prepared_data/test.csv", index_col="PassengerId")
X_test = torch.tensor(
    test_data.to_numpy(),
    dtype=torch.float32,
)
test_dataset = torch.utils.data.TensorDataset(X_test)

params = params_show()["predict"]
model = mlflow.pytorch.load_model(
    f"models:/{params["model_name"]}/{params["model_version"]}"
)

preds = torch.nn.functional.sigmoid(
    model.predict_step(test_dataset[:][0])
).squeeze().detach().round().type(torch.int32).tolist()

pd.DataFrame(
    {"PassengerId": test_data.index, "Survived": preds},
).to_csv("./subs.csv", index=False)

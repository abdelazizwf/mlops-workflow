import pandas as pd
import torch
from torch.utils.data import TensorDataset

train_data = pd.read_csv("prepared_data/train.csv")
val_data = pd.read_csv("prepared_data/val.csv")

X_train = torch.tensor(
    train_data.drop(columns=["Survived", "PassengerId"]).to_numpy(),
    dtype=torch.float32,
)
y_train = torch.tensor(train_data["Survived"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
train_dataset = TensorDataset(X_train, y_train)

X_val = torch.tensor(
    val_data.drop(columns=["Survived", "PassengerId"]).to_numpy(),
    dtype=torch.float32
)
y_val = torch.tensor(val_data["Survived"].to_numpy(), dtype=torch.float32).unsqueeze(-1)
val_dataset = TensorDataset(X_val, y_val)

torch.save(train_dataset, "tensor_data/train_dataset.pt")
torch.save(val_dataset, "tensor_data/val_dataset.pt")

import re

import numpy as np
import pandas as pd
from dvc.api import params_show
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MaxAbsScaler,
    OneHotEncoder,
)

params = params_show(stages="prepare_data")["data"]

np.random.seed(params["random_seed"])
set_config(transform_output="pandas")

def load_data(path):
    data = pd.read_csv(path, index_col="PassengerId")

    data["Title"] = data["Name"].apply(lambda s: re.search(r".+, ([^\.]*\.).*", s).group(1))
    data["CabinSection"] = data["Cabin"].apply(lambda x: x[0] if x is not np.nan else np.nan)
    data["CabinCount"] = data["Cabin"].apply(lambda x: len(x.split()) if x is not np.nan else 0)

    data = data.drop("Name Ticket Cabin".split(), axis=1)

    return data

train_data = load_data("data/train.csv")

X = train_data.loc[:, train_data.columns != "Survived"]
y = train_data[["Survived"]]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, shuffle=True, test_size=params["val_split"], random_state=params["random_seed"]
)

test = load_data("data/test.csv")

numeric_preprocessor = Pipeline([
    ("imp", KNNImputer(n_neighbors=3)),
    ("scaler", MaxAbsScaler()),
])

preprocessor = ColumnTransformer([
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["Embarked", "Title", "CabinSection", "Sex"]),
        ("numeric_preprocessor", numeric_preprocessor, ["Age", "Fare"])
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("fallback_imputer", SimpleImputer(strategy="most_frequent")),
])

X_train = pipeline.fit_transform(X_train)
X_val = pipeline.transform(X_val)
test = pipeline.transform(test)

train = pd.concat([X_train, y_train], axis=1)
val = pd.concat([X_val, y_val], axis=1)

train.to_csv("prepared_data/train.csv")
val.to_csv("prepared_data/val.csv")
test.to_csv("prepared_data/test.csv")

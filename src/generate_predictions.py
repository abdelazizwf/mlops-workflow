import pandas as pd
from omegaconf import OmegaConf
from sklearn.svm import SVC

conf = OmegaConf.load("./params.yaml")
svc_params = conf.model_params.svc

train_data = pd.read_csv("prepared_data/train.csv")
val_data = pd.read_csv("prepared_data/val.csv")
test_data = pd.read_csv("prepared_data/test.csv", index_col="PassengerId")

train_data = pd.concat([train_data, val_data], axis=0, ignore_index=True)
X = train_data.drop(columns=["PassengerId", "Survived"])
y = train_data["Survived"]

model = SVC(**svc_params)

model.fit(X, y)
preds = model.predict(test_data)

pd.DataFrame(
    {"PassengerId": test_data.index, "Survived": preds},
).to_csv("./subs.csv", index=False)

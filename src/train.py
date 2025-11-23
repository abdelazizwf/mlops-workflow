import pandas as pd
from sklearn.svm import SVC

train_data = pd.read_csv("prepared_data/train.csv")
val_data = pd.read_csv("prepared_data/val.csv")

X_train = train_data.drop(columns=["Survived"])
y_train = train_data["Survived"]

X_val = val_data.drop(columns=["Survived"])
y_val = val_data["Survived"]

model = SVC()

model.fit(X_train, y_train)

train_accuracy = model.score(X_train, y_train)
val_accuracy = model.score(X_val, y_val)

print(f"Train Accuracy = {train_accuracy:.3f}\nValidation Accuracy = {val_accuracy:.3f}")

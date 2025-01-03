import os
import pandas as pd
from sklearn import model_selection

df = pd.read_csv('ML_Template/input/mnist_train.csv')

df["kfold"] = -1
df = df.sample(frac=1).reset_index(drop = True)

kf = model_selection.KFold(n_splits=5)

for fold, (trn_, val_) in enumerate(kf.split(X=df)):
    df.loc[val_, 'kfold'] = fold

df.to_csv("ML_Template/input/mnist_train_folds.csv", index=False)
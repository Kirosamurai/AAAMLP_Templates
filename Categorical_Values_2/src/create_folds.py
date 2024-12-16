import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    skf = model_selection.StratifiedKFold(n_splits=5)

    df = pd.read_csv("../input/adult.csv")
    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)

    y = df.income.values

    for f_, (t_, v_) in enumerate(skf.split(X=df, y=y)):
        df.loc[v_, "kfold"] = f_

    df.to_csv("../input/adult_folds.csv", index=False)
import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    df = pd.read_csv("../input/adult_folds.csv")
    df = df.fillna("NONE")

    num_cols = [ 
        "fnlwgt", 
        "age", 
        "capital_gain", 
        "capital_loss", 
        "hours_per_week" 
    ]

    df = df.drop(num_cols, axis=1) 

    df["income"] = df["income"].str.strip()
    target_mapping = { 
        "<=50K": 0, 
        ">50K": 1 
    } 
    df["income"] = df["income"].map(target_mapping)

    features = [f for f in df.columns if f not in ("kfold", "income")]

    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna("NONE")

    df_train = df[df["kfold"] != fold].reset_index(drop = True)
    df_valid = df[df["kfold"] == fold].reset_index(drop = True)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat([df_train[features], df_valid[features]], axis=0)
    ohe.fit(full_data[features])

    x_train = ohe.transform(df_train[features])
    x_valid = ohe.transform(df_valid[features])

    model = linear_model.LogisticRegression()

    model.fit(x_train, df_train.income.values)
    valid_preds = model.predict_proba(x_valid)[:, 1]

    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)

    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)

#not working for some reason
#error: input has NaN
#no null values present anywhere, idk

#NaN values in y, extra space in df.income caused error
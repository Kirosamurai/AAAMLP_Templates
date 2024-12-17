import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()
X = data["data"]
col_names = data["feature_names"]
y=data["target"]

df = pd.DataFrame(X, columns=col_names)

df["MedInc_Sqrt"] = df.MedInc.apply(np.sqrt)
print(df.corr())
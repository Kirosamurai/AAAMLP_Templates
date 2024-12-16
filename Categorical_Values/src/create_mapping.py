import pandas as pd

df = pd.read_csv("../input/train.csv")

mapping = {
    "Freezing": 0, 
    "Warm": 1, 
    "Cold": 2, 
    "Boiling Hot": 3, 
    "Hot": 4, 
    "Lava Hot": 5
}

df.loc[:, "ord_2"] = df.ord_2.map(mapping)
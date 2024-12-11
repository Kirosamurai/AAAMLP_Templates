import pandas as pd

df = pd.read_csv('ML_Template/input/mnist_train.csv')

df.fillna(0, inplace=True)
df = df.astype(int)

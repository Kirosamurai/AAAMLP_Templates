import pandas as pd

def main():
    df = pd.read_csv("../input/train/stage_2_train.csv")
    df['target'] = df['EncodedPixels'].apply(lambda x: 1 if x != '-1' else 0)
    output_df = df[['ImageId', 'target']]
    output_df.to_csv("../input/train/train.csv", index=False)

main()
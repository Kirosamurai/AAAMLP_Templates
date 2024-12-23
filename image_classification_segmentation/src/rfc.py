import os

import numpy as np
import pandas as pd

from PIL import Image
from sklearn import ensemble, metrics, model_selection
from tqdm import tqdm
import pydicom


def create_dataset(training_df, image_dir):
    images = []
    targets = []

    for index, row in tqdm(
        training_df.iterrows(),
        total=len(training_df),
        desc="processing_images"
    ):
    
        image_id = row["ImageId"]
        image_path = os.path.join(image_dir, image_id)

        dcm = pydicom.dcmread(image_path + ".dcm")
        image = dcm.pixel_array
        image = np.array(Image.fromarray(image).resize((256, 256), resample=Image.BILINEAR))

        image = image.ravel()

        images.append(image)
        targets.append(int(row["target"]))
    
    images = np.array(images)
    print(images.shape)
    return images, targets

def main():
    csv_path = "../input/train/train.csv"
    image_path = "../input/stage_2_images/"

    df = pd.read_csv(csv_path)

    df["kfold"] = -1

    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'k_fold'] = f

    for fold_ in range(5):
        train_df = df[df["kfold"] != fold_].reset_index(drop=True)
        test_df = df[df["kfold"] == fold_].reset_index(drop=True)

        x_train, y_train = create_dataset(train_df, image_path)
        x_test, y_test = create_dataset(test_df, image_path)

        clf = ensemble.RandomForestClassifier(n_jobs = -1)
        clf.fit(x_train, y_train)

        preds = clf.predict_proba(x_test)[:, 1]

        print(f"FOLD: {fold_}") 
        print(f"AUC = {metrics.roc_auc_score(y_test, preds)}") 
        print("")

main()
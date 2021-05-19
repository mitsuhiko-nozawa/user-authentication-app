import os
from pathlib import Path
import random
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import f1_score, confusion_matrix
try:
    import lycon
except:
    pass


def train_val_split(data_path:Path, seed:int):
    """
    group person and stratify counts of each person
    retuen:
        train_df
        valid_df
    """
    training_datas = ["casia"]
    df = pd.read_csv(data_path / "data.csv")
    df = df[df["data"].isin(training_datas)]
    gdf = df.value_counts("person_id")
    # train_gdf, valid_gdf = train_test_split(gdf.index, test_size=0.2, shuffle=True, random_state=seed, stratify=gdf.values)
    
    skf = StratifiedKFold(shuffle=True, n_splits=25, random_state=seed)
    tr_inds, val_inds = list(), list()
    for tr_ind, val_ind in skf.split(gdf.index, gdf.values):
        tr_inds.append(tr_ind)
        val_inds.append(val_ind)
    train_persons = gdf.index[tr_inds[0]]
    val_persons = gdf.index[val_inds[0]]
    train_df = df[df["person_id"].isin(train_persons)]
    valid_df = df[df["person_id"].isin(val_persons)]
    return train_df, valid_df


def seed_everything(seed=42):
    """
    fix seed to reproducibility of experiment
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    if torch.cuda.is_available(): 
        print("cuda available")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True



def read_image(file_path):
    image = lycon.load(str(file_path))
    return image

def read_image2(file_path):
    image = cv2.imread(str(file_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image




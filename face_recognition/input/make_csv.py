import os
from pathlib import Path
import pandas as pd

"""
create csv file to read input data conveniently

[attributes]
data : the specific of dataset (lfw, msm1m, ...)
path : path to data file
file_name : file name
"""

INPUT_DIR = Path(__file__).parent

def make_lfw():
    DATA_PATH = INPUT_DIR / "lfw"
    persons = os.listdir(DATA_PATH)
    df = pd.DataFrame()
    datas = []
    person_ids = []
    paths = []
    file_names = []
    for person in persons:
        path = DATA_PATH / person
        files = os.listdir(path)
        for file in files:
            datas.append("lfw")
            person_ids.append(person)
            paths.append(path)
            file_names.append(file)
    df = pd.DataFrame()
    df["data"] = datas
    df["person_id"] = person_ids
    df["path"] = paths
    df["file_name"] = file_names
    return df

def make_casia():
    DATA_PATH = INPUT_DIR / "CASIA-WebFace"
    persons = os.listdir(DATA_PATH)
    df = pd.DataFrame()
    datas = []
    person_ids = []
    paths = []
    file_names = []
    for person in persons:
        path = DATA_PATH / person
        files = os.listdir(path)
        for file in files:
            datas.append("casia")
            person_ids.append(person)
            paths.append(path)
            file_names.append(file)
    df = pd.DataFrame()
    df["data"] = datas
    df["person_id"] = person_ids
    df["path"] = paths
    df["file_name"] = file_names
    return df



if __name__ == "__main__":
    print(INPUT_DIR)
    lfw_df = make_lfw()
    print(lfw_df.shape)
    casia_df = make_casia()
    print(casia_df)
    all_df = lfw_df.append(casia_df)
    
    all_df.to_csv(INPUT_DIR / "data.csv", index=False)
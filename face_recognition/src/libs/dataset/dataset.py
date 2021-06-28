import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import albumentations as A
from sklearn.preprocessing import LabelEncoder

from .transforms import get_transforms
from utils import read_image

class FaceRecognitionDataset(Dataset):
    def __init__(self, phase: str, df: pd.DataFrame, transform: A.Compose):
        self.df = df
        self.size = len(df)
        self.phase = phase
        self.image_paths = df["path"].values
        self.image_files = df["file_name"].values
        self.transform = transform
        self.labels = LabelEncoder().fit_transform(df["person_id"].astype("object").to_list())
        
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image_path = Path(self.image_paths[idx])
        image_file = self.image_files[idx]
        label = self.labels[idx]

        image = read_image(image_path / image_file)
        image = self.transform(image=image)["image"]
        
        if self.phase == "train":
            return image, label
        else:
            return image, 0 # dummy object



def get_dataloader(train_dataset: Dataset, val_dataset: Dataset, batch_size: int, num_workers: int):
    trainloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, pin_memory=True, drop_last=False)
    validloader = DataLoader(val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True, drop_last=False)
    return trainloader, validloader
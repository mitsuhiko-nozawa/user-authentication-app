import sys, os
from pathlib import Path
import pandas as pd

from .manager import BaseManager
from utils import seed_everything, train_val_split
from dataset import FaceRecognitionDataset, get_transforms, get_dataloader
from models import FaceRecognitionModel


class Train(BaseManager):
    def __call__(self):
        # cvで画像を分ける
        print("Training")            
        if self.get("train_flag"):
            seed = self.get("seeds")[0]
            train_df, valid_df = train_val_split(
                data_path=self.data_path,
                seed=seed,
            )
            seed_everything(seed) 
            self.train(train_df, valid_df, seed, 0)

    def train(self, train_df, val_df, seed, fold):
        if self.debug:
            train_df = train_df[:self.get("batch_size")+2]
            val_df = val_df[:self.get("batch_size")+2]

        train_dataset = FaceRecognitionDataset(
            "train",
            train_df, 
            get_transforms(self.get("image_size"), self.get("tr_transforms"), self.get("tr_transform_params"))
        )
        val_dataset = FaceRecognitionDataset(
            "valid",
            val_df, 
            get_transforms(self.get("image_size"), self.get("val_transforms"), self.get("val_transform_params"))
        )
        trainloader, validloader = get_dataloader(train_dataset, val_dataset, self.get("batch_size"), self.get("num_workers"))
        self.params["seed"] = seed
        self.params["fold"] = fold
        self.params["n_classes"] = train_df["person_id"].nunique()
        
        model = FaceRecognitionModel(self.params)
        
        model.fit(trainloader, validloader)

        # valid predict
        #save_instances = save_image_instance(Path(self.val_preds_path))
        #model.read_weight()
        #model.predict(validloader, save_instances)

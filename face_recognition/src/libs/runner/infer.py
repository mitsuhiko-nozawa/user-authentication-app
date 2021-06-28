import os
from pathlib import Path
import numpy as np
import pandas as pd
import pickle
from torch.utils.data import DataLoader

from .manager import BaseManager
from utils import seed_everything
from dataset import FaceRecognitionDataset, get_transforms
from models import FaceRecognitionModel
from utils.metrics import calc_auc

class Infer(BaseManager):
    def __call__(self):
        print("Inference")
        if self.get("infer_flag"):
            test_df = pd.read_csv(self.data_path / "data.csv")
            test_df = test_df[test_df["data"] == "lfw"]
            if self.debug:
                test_df = test_df[:500]
              
            test_dataset = FaceRecognitionDataset(
                "test",
                test_df, 
                get_transforms(self.get("image_size"), self.get("val_transforms"), self.get("val_transform_params"))
            )
            testloader = DataLoader(
                test_dataset, 
                batch_size=self.get("batch_size"), 
                num_workers=self.get("num_workers"), 
                shuffle=False, 
                pin_memory=True,
            )
            seed = self.get("seeds")[0]
            fold = 0
            self.params["seed"] = seed
            self.params["fold"] = fold
            self.params["n_classes"] = test_df["person_id"].nunique()


            model = FaceRecognitionModel(self.params)            
            model.read_weight(f"{seed}_{fold}.pt")
            embs = model.predict(testloader)
            #persons = test_df["person_id"].to_list()
            persons = test_dataset.labels
            test_auc = calc_auc(embs, persons) 
            with open(self.WORK_DIR / "test_auc", 'wb') as f:
                pickle.dump(test_auc, f)  

            print(test_auc) 


                
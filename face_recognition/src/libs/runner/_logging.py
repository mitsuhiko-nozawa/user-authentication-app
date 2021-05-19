import os
import os.path as osp
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
import cv2
import lycon

from .manager import BaseManager
from utils.metrics import calc_auc
from utils import read_image
from utils_torch import inference_fn
from models import FaceRecognitionModel



class Logging(BaseManager):
    def __call__(self):
        print("Logging")
        if self.get("log_flag"):
            if self.get("calc_test_score"): 
                self.calc_test_score()
            if self.get("mlflow"):
                self.create_mlflow()

    def calc_test_score(self):
        with open(self.WORK_DIR / "test_auc", 'rb') as f:
            self.test_auc = pickle.load(f)
         



    def create_mlflow(self):
        print("mlflow")
        import mlflow
        from mlflow.utils.mlflow_tags import MLFLOW_RUN_NAME
        mlflow.set_tracking_uri(str(self.ROOT / "src" / "mlflow" / "mlruns"))
        mlflow.set_experiment("face_recognition")
        with mlflow.start_run():
            mlflow.set_tag(MLFLOW_RUN_NAME, self.get("exp_name"))

            mlflow.log_param("model", self.get("model"))
            mlflow.log_param("image size", self.get("image_size"))
            mlflow.log_param("seeds", self.get("seeds"))


            mlflow.log_metric("test_auc", self.test_auc)

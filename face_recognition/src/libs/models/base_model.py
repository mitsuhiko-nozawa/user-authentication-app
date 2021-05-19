import re
from abc import ABCMeta, abstractmethod
import sys, os
import os.path as osp
from pathlib import Path

import torch

class BaseModel(metaclass=ABCMeta):
    def __init__(self, params=None):
        self.params = params
        self.perse_params()
        self.model = self.get_model(self.params["model"])

    
    @abstractmethod
    def fit(self, train_X, train_y, valid_X, valid_y):
        raise NotImplementedError
    
    @abstractmethod
    def get_model(self):
        raise NotImplementedError

    @abstractmethod   
    def predict(self, X):
        raise NotImplementedError

    @abstractmethod
    def save_weight(self, path):
        raise NotImplementedError

    @abstractmethod
    def read_weight(self, path):
        raise NotImplementedError 

    @abstractmethod
    def perse_params(self):
        raise NotImplementedError 

    def perse_params(self):
        self.ROOT = self.params["ROOT"]
        self.WORK_DIR = self.params["WORK_DIR"]
        self.weight_path = str(Path(self.WORK_DIR) / "weight")
        self.log_path = str(Path(self.WORK_DIR) / "log")
        if not osp.exists(self.log_path): os.mkdir(self.log_path)

        self.n_classes = self.params["n_classes"]
        self.device = torch.device(self.params["device"] if torch.cuda.is_available() else 'cpu')
        self.epochs = self.params["epochs"]
        self.early_stopping_steps = self.params["early_stopping_steps"]
        self.verbose = self.params["verbose"]
        self.seed = self.params["seed"]
        self.fold = self.params["fold"]
        self.accum_iter = self.params["accum_iter"]

        self.loss_tr = self.params["loss_tr"]
        self.loss_fn = self.params["loss_fn"]

        self.optimizer = self.params["optimizer"]
        self.optimizer_params = self.params["optimizer_params"]
        self.scheduler = self.params["scheduler"]
        self.scheduler_params = self.params["scheduler_params"]
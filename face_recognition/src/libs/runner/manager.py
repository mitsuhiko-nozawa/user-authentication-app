from abc import ABCMeta, abstractmethod
import os
from pathlib import Path


class BaseManager(metaclass=ABCMeta):
    def __init__(self, params):
        self.params = params
        self.ROOT = Path(params["ROOT"])
        self.WORK_DIR = Path(params["WORK_DIR"])
        self.data_path = self.ROOT / "input"
        self.val_preds_path = self.WORK_DIR / "val_preds"
        self.preds_path = self.WORK_DIR / "preds"
        self.weight_path = self.WORK_DIR / "weight"

        self.seeds = params["seeds"]
        self.debug = params["debug"]
        self.device = params["device"]
        self.model = params["model"]

        self.env = params["env"]


        if not os.path.exists(self.val_preds_path): os.mkdir(self.val_preds_path)
        if not os.path.exists(self.weight_path): os.mkdir(self.weight_path)
        if not os.path.exists(self.preds_path): os.mkdir(self.preds_path)

        if self.debug:
            self.params["image_size"] = 256
            self.params["batch_size"] = 10
            self.params["epochs"] = 1
            self.params["mlflow"] = False
            self.params["num_workers"] = 0
            self.params["run_folds"] = [0]


    def get(self, key):
        try:
            return self.params[key]
        except:
            raise ValueError(f"No such value in params, {key}")

    @abstractmethod
    def __call__(self):
        raise NotImplementedError
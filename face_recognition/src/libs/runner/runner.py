from .train import Train
from .infer import Infer
from ._logging import Logging


class Runner():
    """
    全体の工程
    ・特徴量作成
    ・データの読み出し
    ・学習、weightと特徴量の名前の保存
    ・ログ(mlflow, feature_importances, )
    """
    def __init__(self, param):
        self.param = param

    def __call__(self):
        Trainer = Train(self.param)
        Inferer = Infer(self.param)
        Logger = Logging(self.param)
        Trainer()
        Inferer()
        Logger()
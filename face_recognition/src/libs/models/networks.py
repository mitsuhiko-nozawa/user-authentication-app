import torch
import torch.nn as nn
import timm
from .arcface import ArcMarginProduct
#import segmentation_models_pytorch as smp


class resnet50d(nn.Module):
    def __init__(self, n_classes, embedding_size=512, s=64.0, m=0.50, enc_name="resnet50d", enc_weights="imagenet", ):
        super().__init__()
        self.model = timm.create_model(enc_name, pretrained=True)
        self.model.global_pool = nn.Sequential(
                                   nn.AdaptiveAvgPool2d(1),
                                   nn.Flatten()
                                )
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, embedding_size)
        self.cosine_softmax = ArcMarginProduct(embedding_size, n_classes, s, m)
    def forward(self, inputs, labels):
        features = self.model(inputs)
        out = self.cosine_softmax(features, labels)
        return out


from albumentations import (
    HorizontalFlip, VerticalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90, Solarize,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine, RandomResizedCrop,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, Normalize, Cutout, CoarseDropout, ShiftScaleRotate, CenterCrop, Resize
)
from albumentations.pytorch import ToTensorV2
from albumentations import ImageOnlyTransform, DualTransform
import cv2
import numpy as np

def get_transforms(img_size, transforms, params):    
    params["size"] = img_size
    return Compose([ get_aug(aug, params) for aug in transforms])


def get_aug(aug, params):
    if aug in ["Resize"]:
        return eval(aug)(params["size"], params["size"])

    elif aug in ["RandomResizedCrop"]:
        return eval(aug)(params["size"], params["size"], scale=(0.5, 1.0))
    
    elif aug in ["RandomResizedCrop2"]:
        return RandomResizedCrop(params["size"], params["size"], scale=(0.2, 1.0))

    elif aug in ["Transpose", "HorizontalFlip", "VerticalFlip", "ShiftScaleRotate"]:
        return eval(aug)(p=0.5)

    elif aug in ["HueSaturationValue"]:
        return eval(aug)(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5)

    elif aug in ["RandomBrightnessContrast"]:
        return eval(aug)(brightness_limit=(-0.1,0.1), contrast_limit=(-0.1, 0.1), p=0.5)

    elif aug in ["Solarize"]:
        return eval(aug)((230, 255), p=1.)

    elif aug in ["Normalize"]:
        return eval(aug)(mean=params["color_mean"], std=params["color_std"])

    elif aug in ["CoarseDropout"]:
        return eval(aug)(p=0.2)

    elif aug in ["Cutout"]:
        return eval(aug)(p=0.2, fill_value=255, num_holes=50, max_h_size=10, max_w_size=10)

    elif aug in ["ToTensorV2"]:
        return eval(aug)()

    elif aug in ["Scale"]:
        return eval(aug)(params["scaleIn"])


class Scale(DualTransform):
    def __init__(self, scale=1):
        super(Scale, self).__init__(always_apply=True, p=1.0)
        self.scale = scale # original images are 2048 x 1024

    def apply(self, img, **params):
        # apply to image
        return img
    def apply_to_mask(self, mask, **params):
        # apply to mask
        h, w = mask.shape[:2]
        mask = cv2.resize(mask, (int(w/self.scale), int(h/self.scale)), interpolation=cv2.INTER_NEAREST)
        return mask

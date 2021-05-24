import sys, os
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings("ignore")
from omegaconf import DictConfig, OmegaConf
import cv2
import time
import numpy as np
import torch
from torchsummary import summary
import psutil

ROOT = Path(os.getcwd()) # ~/11_demospace

### app contents ###
process = psutil.Process(os.getpid())
print(f"[ init ] memused : {process.memory_info().rss / 1024 / 1024} MB", )

# カメラ
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
IMG_SIZE = 512

# Back Ground 


"""
一回fp16にしてからfp32に戻すと軽くなる(精度への影響もある)
ボトルネックになってる二回目のdilatedのところだけでもいいかも
"""



cnt = 0
s = time.time() # second

def read_image(cap):
    ret, img = cap.read()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.flip(img, 1)
    cv2.imshow('', img)
while cap.isOpened:
    #ret, img = cap.read()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.flip(img, 1)
    #cv2.imshow('', img)
    read_image(cap)
    # メモリ使用量を取得 
    cnt += 1
    t = time.time() - s
    
    print(f"[runing] memused : {process.memory_info().rss / 1024 / 1024} MB | fps : {cnt / t}")


    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

print("program was terminated by q key")
cap.release()
cv2.destroyAllWindows()

import cv2
from mtcnn import MTCNN
import lycon
import pandas as pd

detector = MTCNN()
df = pd.read_csv("/workspace/face_recognition/input/data.csv")
img_paths = [path + "/" + file for path, file in zip(df["path"].to_list(),df["file_name"].to_list())][:10]
for img_path in img_paths:
    img = lycon.load(img_path)
    results = detector.detect_faces(img)

    result = None
    conf = -1
    for res in results:
        if conf <= res["confidence"]:
            result = res
    

    print(result["box"])
    x, y, w, h = result["box"]
    res_img = img[y:y+h, x:x+w]
    cv2.imwrite("/workspace/face_recognition/input/temp.jpeg", cv2.cvtColor(res_img, cv2.COLOR_RGB2BGR))
    

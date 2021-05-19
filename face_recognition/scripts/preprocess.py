from facenet_pytorch import MTCNN
import pandas as pd
import torch
import lycon

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
print(device)
mtcnn = MTCNN(image_size=112, device=device)
df = pd.read_csv("/workspace/face_recognition/input/data.csv")
img_paths = [path + "/" + file for path, file in zip(df["path"].to_list(),df["file_name"].to_list())]
for img_path in img_paths:
    save_path = img_path.replace("/lfw/", "/lfw_prepro/").replace("CASIA-WebFace", "CASIA-WebFace_prepro")
    img = lycon.load(img_path)
    try:
        img_cropped = mtcnn(img, save_path=save_path)
    except:
        print(save_path)
        pass

    

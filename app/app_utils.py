import sys, os
import time
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn.functional as F
import streamlit as st
import albumentations as A
from albumentations.pytorch import ToTensorV2
import mediapipe as mp
from facenet_pytorch import MTCNN




def user_registration(ROOT, cap, image_loc, pic, input_userid):
    phase = "picture"
    sub_header = st.subheader("")

    # picture phase
    cnt = 0
    
    # password phase
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    circle_mask = create_circle_mask(W, H) # (H, W)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    passwords = [-1]
    is_existingHands = False
    save_image = None

    while cap.isOpened:
        ret, image = cap.read()
        if not ret:
            continue
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        if phase == "picture":
            if pic:
                cnt += 1
                if cnt >= 15:
                    # face crop
                    save_image = image
                    face_cropping(save_image, str(ROOT/"temp.jpg"))
                    #cv2.imwrite(str(ROOT/"temp.jpg"), cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR))
                    phase = "password"
                    sub_header.subheader("took picture!")
                    time.sleep(3)
                    sub_header.subheader("password")

        elif phase == "password":
            with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                results = hands.process(image)
                # 人差し指の座標
                if results.multi_hand_landmarks:
                    if not is_existingHands:
                        is_existingHands = True
                    finger = results.multi_hand_landmarks[0].landmark[8]
                    selected_circle = circle_mask[int(H * finger.y)][int(W * finger.x)]
                    draw_circle(image, selected_circle)
                
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    if passwords[-1] != selected_circle and selected_circle != 0:
                        passwords.append(int(selected_circle))
                    print(passwords)
                    
                else:
                    draw_circle(image, 0)
                    if is_existingHands and len(passwords) >= 5: # terminate password phase
                        phase = "done"
                        sub_header.subheader("done")
                                                
        elif phase == "done":
            save_user(ROOT, input_userid, save_image, passwords[1:])
            break

        image_loc.image(image)


def draw_circle(image, selected_circle):
    H, W = image.shape[:2]
    H_c, W_c = H // 2, W // 2
    q = H // 4
    rad = 50
    cnt = 1
    color = (192, 192, 192)
    for i in range(-1, 2):
        for j in range(-1, 2):
            if cnt == selected_circle:
                cv2.circle(image, (W_c+j*q, H_c+i*q-q//2), rad, (117, 175, 59), thickness=-1)
            else:
                cv2.circle(image, (W_c+j*q, H_c+i*q-q//2), rad, color, thickness=-1)
            cv2.putText(image, str(cnt), (W_c+j*q, H_c+i*q-q//2), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), thickness=2)
            cnt += 1


def create_circle_mask(W, H):
    mask = np.zeros((H, W))
    H_c, W_c = H // 2, W // 2
    q = H // 4
    cnt = 1
    rad = 50
    for i in range(-1, 2):
        for j in range(-1, 2):
            cv2.circle(mask, (W_c+j*q, H_c+i*q-q//2), rad, (cnt), thickness=-1)
            cnt += 1
    return mask

    
def save_user(ROOT: Path, user_id: str, save_image: np.array, passwords: list):
    os.makedirs(ROOT / "images" / user_id, exist_ok=True)
    save_path = str(ROOT / "images" / user_id / "000.jpg")
    passwords = "".join([str(v) for v in passwords])
    image_paths_df = pd.read_csv(ROOT / "image_paths.csv")
    user_passwords_df = pd.read_csv(ROOT / "user_passwords.csv")
    image_paths_df = image_paths_df.append({"user_id": user_id, "image_path" : save_path }, ignore_index=True)
    user_passwords_df = user_passwords_df.append({"user_id" : user_id, "password" : passwords}, ignore_index=True)
    
    face_cropping(save_image, save_path)
    image_paths_df.to_csv(ROOT / "image_paths.csv", index=False)
    user_passwords_df.to_csv(ROOT / "user_passwords.csv", index=False)
    st.subheader(f"your user name: {user_id}")
    st.subheader(f"your password: {passwords}")



class Authentication:
    def __init__(self, ROOT):
        self.ROOT = ROOT
        self.phase = "face matching"
        self.model = build_model(ROOT)
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.similarity = None
        self.user = None
        self.user_password = None
        self.transforms = A.Compose([
            A.Normalize(),
            ToTensorV2()
        ])

    def get_embedding(self, image):
        image = self.transforms(image=image)["image"]
        return F.normalize(self.model(image.unsqueeze(0))).squeeze(0).detach().cpu().numpy()
        


    def face_matching(self, image):
        face_cropping(image, str(self.ROOT / "input.jpg"))
        image = cv2.imread(str(self.ROOT / "input.jpg"))

        df = pd.read_csv(self.ROOT / "image_paths.csv")
        users = df["user_id"].to_list()
        image_paths = df["image_path"].to_list()
        passwords = pd.read_csv(self.ROOT / "user_passwords.csv")["password"].astype("str").to_list()
        input_emb = self.get_embedding(image)
        similarities = []
        for image_path in image_paths:
            pair_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
            embedding = self.get_embedding(pair_image)
            similarities.append(np.dot(input_emb, embedding))
        print(similarities)
        
        most_similar_idx = np.argmax(similarities)
        if similarities[most_similar_idx] >= 0.4:
            self.similarity = similarities[most_similar_idx]
            self.user = users[most_similar_idx]
            self.user_password = passwords[most_similar_idx]

        self.phase = "password"


    def authenticate(self, cap, image_loc):
        W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        circle_mask = create_circle_mask(W, H) # (H, W)
        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands
        passwords = [-1]
        header_one = st.subheader("")
        header_two = st.subheader("")
        s = time.time()

        while cap.isOpened:
            ret, image = cap.read()
            if not ret:
                continue
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

            if self.phase == "face matching":  
                image_loc.image(image)
                elapsed_time = time.time() - s
                remaining_time = 4-int(elapsed_time)
                header_one.subheader(f"{remaining_time}")
                if remaining_time <= 0:
                    self.face_matching(image)

            elif self.phase == "password" and self.user is not None:
                with self.mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
                    results = hands.process(image)
                    # 人差し指の座標
                    if results.multi_hand_landmarks:
                        finger = results.multi_hand_landmarks[0].landmark[8]
                        selected_circle = circle_mask[int(H * finger.y)][int(W * finger.x)]
                        draw_circle(image, selected_circle)
                
                        for hand_landmarks in results.multi_hand_landmarks:
                            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        if passwords[-1] != selected_circle and selected_circle != 0:
                            passwords.append(int(selected_circle))
                        
                    else:
                        draw_circle(image, 0)
                    header_one.subheader(f"hello {self.user}, your input : {''.join([str(ch) for ch in passwords[1:]])}")
                    header_two.subheader(f"similarity : {self.similarity}")
                    

                    if "".join([str(ch) for ch in passwords[1:]]) == self.user_password:
                        image_loc.image(image)
                        header_two.subheader("password matching!")
                        self.phase = "done"
                        time.sleep(1)
                        

                image_loc.image(image)
            else :
                if self.user is None:
                    header_one.subheader("you are not registered")
                #image_loc.image(np.ones((1, 1)))
                break
            
def build_model(ROOT):
    #ROOT.parent / "face_recognition" / "src" / "libs"
    sys.path.append(str(ROOT.parent / "face_recognition" / "src" / "libs"))
    from models import resnet50d
    device = "cpu"
    model = resnet50d(n_classes=10).model
    #model.load_state_dict(torch.load( str(ROOT / "weight" / "2021_0.pt") , map_location=device), device)
    model.load_state_dict(torch.load( str(ROOT.parent / "face_recognition" / "src" / "experiments" / "exp_002" / "weight"/"2021_0.pt") , map_location=device), device)
    model.eval()
    return model


def face_cropping(image, save_path):
    mtcnn = MTCNN(image_size=112)
    image_cropped = mtcnn(image, save_path)
    return 
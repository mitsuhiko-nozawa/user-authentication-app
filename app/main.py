import streamlit as st
import cv2
from PIL import Image
import mediapipe as mp
import time
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

st.title('Virtual Background Application')
cap = cv2.VideoCapture(0)

image_loc = st.empty()
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened:
        t = time.time()
        ret, image = cap.read()
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)

        image.flags.writeable = True
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        image_loc.image(image)
        s = time.time() - t

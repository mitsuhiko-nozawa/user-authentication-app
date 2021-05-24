import streamlit as st
import cv2
from PIL import Image
import time
from pathlib import Path

from app_utils import *



ROOT = Path(__file__).parent # ~/app

st.title('Virtual Background Application')
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 10)

sub_header = st.subheader("")
image_loc = st.empty()
text_time = "[fps] : %f"
time_disp = st.sidebar.empty()


### 

st.sidebar.subheader("Registration Section")
input_userid = st.sidebar.text_input("input your user_id")
do_register = st.sidebar.checkbox("Register")
pic = st.sidebar.checkbox("Take a picture")

st.sidebar.markdown("---")
st.sidebar.subheader("Authentication Section")
do_authntication = st.sidebar.checkbox("Authentication")

Auth = Authentication(ROOT)

if pic : 
    do_register = True



if do_register:
    if len(input_userid) == 0:
        st.sidebar.write('<span style="color:red;background:pink">[warning] please input your use_id</span>', unsafe_allow_html=True)
    else:
        print(input_userid)
        sub_header.subheader("registration phase")
        user_registration(ROOT, cap, image_loc, pic, input_userid)
        time.sleep(1)
        image_loc.image(np.ones((1, 1)))

if do_authntication:
    Auth.authenticate(cap, image_loc)

    


print("fin")
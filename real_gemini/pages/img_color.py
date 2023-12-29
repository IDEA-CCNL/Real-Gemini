import cv2
import streamlit as st
img = st.camera_input('tt',label_visibility='hidden')
if img:
    st.image(img)
    print(type(img))
    print(img)
capture = cv2.VideoCapture(0)
ret,frame = capture.read()
st.text('摄像头捕捉：')
st.image(frame)
# print(frame[1].shape)
st.text('frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)')
frame1 = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
st.image(frame1)
st.text("cv2.imencode('.jpg', img)")
frame2 = cv2.imencode('.png', frame1)[1]
frame3 = cv2.imdecode(frame2,cv2.COLOR_BGR2RGB)
st.image(frame3)
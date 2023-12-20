import streamlit as st
import speech_recognition as sr 
import base64
from utils.text2audio import text2audio

if st.button('录音'):
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write('请开始说话')
        audio = r.listen(source)
        st.write('监听了')
        st.audio(audio)
        try:
            text = r.recognize_bing(audio,language='zh')
            st.write('你刚刚说的：',text)
        except:
            st.write('语音识别失败')

audio,rate = text2audio('你好')
with open('source/test.m4a','rb') as f:
    audio_b64 = base64.b64encode(f.read())
my_u = base64.b64decode(audio_b64)
st.audio(audio,sample_rate=rate)
st.audio('source/test.m4a')
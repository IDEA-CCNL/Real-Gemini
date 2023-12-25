import streamlit as st
import speech_recognition as sr 
import base64
from utils.audio2text import audio2text_from_bytes
from moviepy.editor import AudioFileClip
from utils.record_video import VideoRecorder
import time
from utils.record_video import record


if __name__ == '__main__':
    st.camera_input('tt',label_visibility='hidden')
    imgs,audio = None,None
    if st.button('开始对话'):
        imgs,audio = record()
        text = audio2text_from_bytes(audio.get_wav_data())
        st.audio(audio.get_wav_data())
        st.write('你刚刚说的：',text)
        st.write('下面是同步录制的图片')
        # for i in range(10):
        #     st.image(imgs.get()) #queue
        for img in imgs:
            st.image(img)
    # except Exception as e:
    #     print(e)
    #     st.write('语音识别失败')
    

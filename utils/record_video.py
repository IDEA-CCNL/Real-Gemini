import time
import streamlit as st
from queue import Queue
import cv2
from datetime import datetime
import time
from threading import Thread,Event
import speech_recognition as sr 
from utils.audio2text import audio2text_from_bytes

class VideoRecorder():
    def __init__(self,record_fps=0.5,max_record_time=60):
        self.frames = list()
        self.max_record_time = max_record_time
        self.record_fps = record_fps
        self.stop_singl = False
        self.process = Thread(target=self.record_v_a)
        self.exit = Event()

    def record_v_a(self):
        print('开始录制')
        self.capture = cv2.VideoCapture(0)
        self.capture.set(3,640) # with
        self.capture.set(4,480) # hight
        s_t = time.time()
        img_id = 0
        while(True):
            if (time.time()-s_t) % self.record_fps == 0:
                try:
                    ret, frame = self.capture.read()
                except:
                    print('error～')
                    break
                if ret:
                    img_id += 1 
                    print(f'正常进入 ret：{ret},img_id:{img_id}')
                    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    # st.image(frame)
                    self.frames.append(frame)
                    # self.frames.put(frame)
                else:
                    print('captrue closed')
                    break
                if self.exit.is_set():
                    print('手动结束')
                    break
                if (time.time()-s_t) > self.max_record_time:
                    print('超时退出')
                    break
        self.capture.release()
        print('录制结束')

    def stop_record(self):
        self.exit.set()
        print('手动杀死线程')
        

def record():
    r = sr.Recognizer()
    r.energy_threshold=500 # 检测声音的阈值
    with sr.Microphone() as source:
        video_record = VideoRecorder()
        st.write('请开始说话，下面开始监听') 
        # phrase_time_limit 最大录制时常，timeout 等待时常
        video_record.process.start()# 这个会进行录像
        audio = r.listen(source,phrase_time_limit=15,timeout=3)
        time.sleep(3) # 额外往后录制2秒钟
        video_record.stop_record()
    return video_record.frames,audio

if __name__ == "__main__":
    # https://blog.csdn.net/qq_42069296/article/details/133792896
    if st.button('开始录制'):
        st.camera_input('tt',label_visibility='hidden')
        recorder = VideoRecorder()
        recorder.process.start()
        time.sleep(10)
        recorder.stop_record()
        print(recorder.frames)
        for _ in range(10):
            st.image(recorder.frames.get())
        print('ok~')


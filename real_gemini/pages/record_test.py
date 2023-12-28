import time
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from queue import Queue
import cv2
from datetime import datetime
import time
from threading import Thread,Event

class VideoRecorder():
    def __init__(self,record_fps=0.5,max_record_time=10):
        self.frames = Queue(120)
        self.max_record_time = max_record_time
        
        self.record_fps = record_fps
        self.stop_singl = False
        self.process = Thread(target=self.record_v_a)
        self.exit = Event()

    def record_v_a(self):
        print('开始录制')
        self.capture = cv2.VideoCapture(0)
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
                    # frame.tobytes()
                    # st.image(frame)
                    self.frames.put(frame)
                else:
                    print('captrue closed')
                    break
                if self.exit.is_set():
                    print('手动结束')
                    break
                if (time.time()-s_t) > 60:
                    print('超时退出')
                    break
        self.capture.release()
        print('录制结束')
                    
    
    def stop_record(self):
        self.exit.set()
        print('手动杀死线程')
        


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


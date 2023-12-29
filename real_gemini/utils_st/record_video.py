import cv2
import time
import streamlit as st
import speech_recognition as sr 
from threading import Thread,Event

class VideoRecorder():
    # 基于这个修改https://blog.csdn.net/qq_42069296/article/details/133792896
    def __init__(self,record_fps=0.5,max_record_time=60):
        self.frames = list()
        self.max_record_time = max_record_time
        self.record_fps = record_fps
        self.stop_singl = False
        # https://blog.csdn.net/captain5339/article/details/128360804
        self.process = Thread(target=self.record_v_a)
        self.exit = Event()

    def record_v_a(self):
        print('开始录制')
        # https://blog.csdn.net/weixin_40922744/article/details/103356458
        self.capture = cv2.VideoCapture(0)
        # 设置分辨率
        # self.capture.set(3,640) # with
        # self.capture.set(4,480) # hight
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
                    img_id += 1 #记录录的帧
                    # https://www.jianshu.com/p/0e462b4c7a93
                    # 不能在这里做转换，否则gpt4v的接口识别到的颜色是反过来的，就是说接口那边会再做一次转换
                    # frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                    self.frames.append(frame)
                else:
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
    # https://blog.51cto.com/u_16213389/7407010
    r = sr.Recognizer()
    # https://blog.csdn.net/sunriseYJP/article/details/134399727
    r.energy_threshold=500 # 检测声音的阈值
    with sr.Microphone() as source:
        video_record = VideoRecorder()
        st.write('请开始说话，下面开始监听') 
        # phrase_time_limit 最大录制时常，timeout 等待时常
        video_record.process.start()# 这个会进行录像
        audio = r.listen(source,phrase_time_limit=15,timeout=None)
        # time.sleep(2) # 额外往后录制2秒钟 监听本身带了延时，无需再加
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


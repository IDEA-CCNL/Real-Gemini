import streamlit as st
import speech_recognition as sr 
import base64
from utils_st.audio2text import audio2text_from_bytes
from moviepy.editor import AudioFileClip
from utils_st.record_video import VideoRecorder
import time
from utils_st.record_video import record
from queue import Queue

max_turn = 20
q = Queue(max_turn)

def audio_record():
    r = sr.Recognizer()
    r.energy_threshold=500 # 检测声音的阈值
    with sr.Microphone() as source:
        st.write('请开始说话，下面开始监听') 
        # phrase_time_limit 最大录制时常，timeout 等待时常
        for i in range(max_turn):
            print(f'turn {i} start')
            audio = r.listen(source,phrase_time_limit=15,timeout=None)
            q.put(audio)
            print(f'声音录制结束，{q.qsize()}')
    print('结束')

def my_recorder():
    for i in range(max_turn):
        imgs,audio = record()
        q.put((imgs,audio))
        print(f'录制结束，{q.qsize()}')
    print('输入处理服务结束')

def res():
    print('getin response')
    i = 20
    while i>0:
        if q.empty():
            print(f'q is empty , waiting')
            time.sleep(5)
            i-=1
        else:
            print('reqeusts ok~')
            audio = q.get()
            st.audio(audio.get_wav_data())
            input_text,code_status,request_id = audio2text_from_bytes(audio.get_wav_data())
            print(f'这是识别出来的文字：{input_text}')
            st.text(f'这是识别出来的文字：{input_text}')
            i-=1
    print('response over~')

def show_chat_message_from_history():
    pass

if __name__ == '__main__':
    from threading import Thread
    t1 = Thread(target=my_recorder)
    t2 = Thread(target=res)
    # st.camera_input('tt',label_visibility='hidden')
    st.camera_input('tt',label_visibility='hidden')
    if st.button('开始对话'):
        t1.start()
        # t2.start()
        # t1.join()
        # t2.join()
        i = 20
        placeholder = st.empty()
        while i>0:
            if q.empty():
                print(f'q is empty , waiting')
                time.sleep(5)
                i-=1
            else:
                print('reqeusts ok~')
                imgs,audio = q.get()
                st.audio(audio.get_wav_data())
                input_text,code_status,request_id = audio2text_from_bytes(audio.get_wav_data())
                print(f'这是识别出来的文字：{input_text}')
                st.text(f'这是识别出来的文字：{input_text}')
                st.text('下面是录制到的图片')
                for idx,cl in enumerate(st.columns(min(3,len(imgs)))):
                    cl.image(imgs[idx])
                st.divider()
                i-=1
        print('response over~')

    

import uuid
import streamlit as st
from pathlib import Path
from utils_st.audio2text import audio2text_from_bytes
from utils_st.extracte_img import get_main_img
from utils_st.get_gpt4v_response import gpt4v_client
from utils_st.get_qwen_response import QwenVL_client
from utils_st.text2audio import text2audio,autoplay_audio
from utils_st.record_video import record
from queue import Queue
import time
import cv2
from threading import Thread,Event

img = {'assistant':'./source/bot.png','user':None}
res_ = {'Qwen-vl':QwenVL_client,'gpt4v':gpt4v_client}

# 设置事件锁
event_record = Event()
event_chat = Event()
event_record.set() # 初始打开录音锁

with st.sidebar:
    with st.form('参数配置'):
        max_chat_turn = st.slider('最大对话轮数:',min_value=1,max_value=10000,value=10)
        response_name = st.selectbox('选择模型',['Qwen-vl','gpt4v'],index=1)
        st.form_submit_button('提交配置')
responser = res_[response_name]
max_record_round = 2*max_chat_turn
q = Queue(max_record_round)

st.title("Gemini-like对话测试")
#########################存储录入的文件#####################
# RECORD_DIR = Path("./records")
# RECORD_DIR.mkdir(exist_ok=True)
# if "prefix" not in st.session_state:
#     st.session_state["prefix"] = str(uuid.uuid4())
# prefix = st.session_state["prefix"]
# in_file_video = RECORD_DIR / f"{prefix}_input_video.mp4"
# in_file_audio = RECORD_DIR / f"{prefix}_input_audio.mp3"
#########################存储录入的文件#####################
# 对话机器人的图标


if "messages" not in st.session_state:
    st.session_state.messages = []

def my_recorder():
    for i in range(max_record_round):
        # 等待录入条件触发，最开始是默认触发
        print('holding to record')
        event_record.wait()
        print(f'record {i}')
        imgs,audio = record()
        input_text,code_status,request_id = audio2text_from_bytes(audio.get_wav_data())
        # 过滤一些无意义的文本
        if input_text and len(input_text)>5:
            q.put((imgs,audio,input_text))
        else:
            print(f'非预期输入: id--{request_id},status--{code_status},text--{input_text}')
            time.sleep(2)# 给2秒时间，调整准备输入
            continue
        print(f'{i}录制结束，{q.qsize()}')
        # 录制结束，解开对话阻塞，同时阻塞下一轮录入
        event_record.clear()
        event_chat.set()
        print('释放对话锁，加录音锁')
    print('输入处理服务结束')

def show_chat_message_from_history(show_num_history=None):
    # Display chat messages from history on app rerun
    # show_num_history: 应当为负偶数或者正奇数，负偶数表示为最后N条，正数表示跳过前N条
    if show_num_history is None:
        history = st.session_state.messages
    else:
        history = st.session_state.messages[show_num_history:]
    for message in history:
        with st.chat_message(message["role"],avatar=img[message['role']]):
            try:
                if message['audio'] is not None:
                    st.audio(message['audio'],sample_rate=24000)
            except:
                pass
            st.markdown(message["content"])
            try:
                if message['img'] is not None:
                    st.image(message['img'])
            except:
                pass

def response(prompt=None,imgs=None,autoplay=True,audio_response=True):
    """
    prompt：输入的文本
    imgs：输入的图片
    autoplay：是否自动播放语音
    audio_response：是否将文本转换成语音响应
    """
    if prompt:
        sound = None
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
        # Display assistant response in chat message container
        with st.chat_message("assistant",avatar='./source/bot.png'):
            res = responser(query=prompt,imgs=imgs)
            print('res[text]:',res['text'])
            if audio_response:
                sound,rate,byte_sound_array = text2audio(res["text"])
            else:
                autoplay = False
            if autoplay:
                autoplay_audio(byte_sound_array)
            if not autoplay and audio_response:
                # 不自动播放语音
                st.audio(sound,sample_rate=rate)
            st.markdown(res['text'])
            try:
                st.image(res['imgs'])
            except:
                pass
            # 由于是自动播放音频，需要等待音频播放完毕
            if autoplay:
                time.sleep(int(len(sound)/rate)+1)
            st.session_state.messages.append({"role": "assistant", "content": res['text'],'audio':sound})


if __name__ == '__main__':
    max_round=max_chat_turn+50 # 为了保证安全，没有写没条件的while循环
    record_thread = Thread(target=my_recorder)
    # 展示录像设备的图像信息
    video_show = st.container()
    video_show.camera_input('tt',label_visibility='hidden')
    # 开始录入输入
    if video_show.button('开始对话'):
        st.info(f'开始监听麦克风...')
        record_thread.start()
    else:
        st.stop()
    # 展示录入信息的处理
    placeholder = st.empty()
    # 展示对话
    chat_placeholder = st.empty()
    while max_round>0:
        # 等待对话开始，初始化是阻塞，等待第一次输入录入完成，才会打开锁
        print('等待对话开始')
        event_chat.wait()
        print('开始对话')
        if not q.empty():
            # 进入到对话时，停止录入，防止录入播放的音频
            print('进入对话响应，暂停录入')
            imgs,audio,input_text = q.get()
            with placeholder.status('处理输入信号...',state='running',expanded=True) as status:
                if len(imgs)>0:
                    st.write('getMainFrame...')
                    imgs = get_main_img(imgs)
                    imgs = imgs[-3:]
                    cls = st.columns(min(3,len(imgs)))
                    for idx,cl in enumerate(cls):
                        cl.image(cv2.cvtColor(imgs[idx],cv2.COLOR_BGR2RGB))
                st.audio(audio.get_wav_data())
                st.text(f'识别后的文本：{input_text}')
                status.update(label="输入信号处理完成", state="complete", expanded=False)
            with chat_placeholder.container(height=600):# 1.30支持设置 height=300px
            # with st.container(height=600):# 1.30支持设置 height=300px
                # 容器高度设置，要等1.30版本更新，https://github.com/streamlit/streamlit/issues/2169
                show_chat_message_from_history() # 现在关闭展示历史，只展示单轮
                response(prompt=input_text,imgs=imgs,autoplay=True,audio_response=True)
                print('对话完毕，释放录音锁，打开对话锁')
                # 对话响应完毕，打开事件
                event_record.set()
                # 如果没有录入输入，等待
                event_chat.clear()
            # chat_placeholder.empty()
    print('达到最大对话轮数，结束程序！')

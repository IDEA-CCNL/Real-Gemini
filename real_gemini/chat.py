import os
import sys
import shutil
import streamlit as st
from queue import Queue
import time
import cv2
from threading import Thread, Event
from .tools.tts_tool import TTSTool
from .utils.audio2text import audio2text_from_bytes
from .utils.image_selector import get_main_img
from .utils.text2audio import autoplay_audio
from .utils.record_video import record
from .agent import ReActAgent

class ChatEngine:
    # 设置事件锁
    event_record = Event()
    event_chat = Event()
    event_record.set() # 初始打开录音锁
    record_round = 100
    # 图片缓存路径
    image_buf_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "img_buf")

    # 初始化agent
    gemini_agent = ReActAgent()
    # 事件队列
    event_queue = Queue(record_round)
    # 资源
    avatar_dict = {'assistant': './source/bot.png', 'user': None}

def record_handler():
    for round in range(ChatEngine.record_round):
        # 等待录入条件触发，最开始是默认触发
        print('holding to record')
        ChatEngine.event_record.wait()
        print(f'record {round}')
        imgs, audio = record()
        input_text, code_status, request_id = audio2text_from_bytes(audio.get_wav_data())
        if input_text and len(input_text)>5:
            ChatEngine.event_queue.put((imgs, audio, input_text))
        else:
            ChatEngine.event_queue.put((imgs, audio, None))
            print(f'非预期输入: id--{request_id}, status--{code_status}, text--{input_text}')
        print(f'{round}录制结束，{ChatEngine.event_queue.qsize()}')
        # 录制结束，解开对话阻塞，同时阻塞下一轮录入
        ChatEngine.event_record.clear()
        ChatEngine.event_chat.set()
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
        with st.chat_message(message["role"], avatar=ChatEngine.avatar_dict[message['role']]):
            if 'audio' in message and message['audio'] is not None:
                st.audio(message['audio'], sample_rate=24000)
            st.markdown(message["content"])
            if 'img' in message and message['img'] is not None:
                st.image(message['img'])

def save_buf_image(imgs):
    if os.path.exists(ChatEngine.image_buf_dir):
        shutil.rmtree(ChatEngine.image_buf_dir)
    os.makedirs(ChatEngine.image_buf_dir)
    for idx, img in enumerate(imgs):
        filename = os.path.join(ChatEngine.image_buf_dir, f"image_{idx}.png")
        cv2.imwrite(filename, img)

def response(prompt=None, imgs=None, autoplay=True, audio_response=True):
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
        with st.chat_message("assistant", avatar='./source/bot.png'):
            save_buf_image(imgs)
            res = ChatEngine.gemini_agent.run(
                prompt=prompt, image_path_or_dir=ChatEngine.image_buf_dir)
            print('res:', res)
            # 如果有图片的话
            if "image" in res:
                st.image(res['image'])
            if res["text"] == "###":
                st.toast("无效的语音输入，请重试。", icon="🤖")
                st.markdown("无效的语音输入")
                res["text"] = "无效的语音输入"
            else:
                if audio_response:
                    tts_tool = TTSTool()
                    sound, rate, byte_sound_array = tts_tool.inference(res["text"])
                else:
                    autoplay = False
                if autoplay:
                    autoplay_audio(byte_sound_array)
                if not autoplay and audio_response:
                    # 不自动播放语音
                    st.audio(sound, sample_rate=rate)
                st.markdown(res['text'])
                # 由于是自动播放音频，需要等待音频播放完毕
                if autoplay:
                    time.sleep(int(len(sound)/rate)+1)
            # 如果有音频的话
            if "audio" in res:
                if autoplay:
                    autoplay_audio(res["audio"])
                    time.sleep(10)
                else:
                    st.audio(res['audio'])
            st.session_state.messages.append(
                {"role": "assistant", "content": res['text'], 'audio': sound})


def launch():
    st.title("Real-Gemini")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    # 展示录像设备的图像信息
    video_show = st.container()
    video_show.camera_input('tt', label_visibility='hidden')
    # 开始录入输入
    record_thread = Thread(target=record_handler)
    if video_show.button('开始对话'):
        record_thread.start()
        st.balloons()
        st.info("现在你可以开始进行连续的多轮对话了...")
    else:
        st.stop()
    # 展示录入信息的处理
    placeholder = st.empty()
    # 展示对话
    chat_placeholder = st.empty()
    while True:
        # 等待对话开始，初始化是阻塞，等待第一次输入录入完成，才会打开锁
        print('等待对话开始')
        ChatEngine.event_chat.wait()
        print('开始对话')
        if not ChatEngine.event_queue.empty():
            # 进入到对话时，停止录入，防止录入播放的音频
            print('进入对话响应，暂停录入')
            imgs, audio, input_text = ChatEngine.event_queue.get()
            if input_text:
                with placeholder.status('处理输入信号...', state='running', expanded=True) as status:
                    if len(imgs) > 0:
                        st.write('获取关键帧...')
                        imgs = get_main_img(imgs, 3)
                        cls = st.columns(min(3, len(imgs)))
                        for idx, cl in enumerate(cls):
                            cl.image(cv2.cvtColor(imgs[idx], cv2.COLOR_BGR2RGB))
                    st.audio(audio.get_wav_data())
                    st.text(f'识别后的文本：{input_text}')
                    status.update(label="输入信号处理完成", state="complete", expanded=False)
                with st.spinner('玩命跑agent中...'):
                    with chat_placeholder.container():# 1.30支持设置 height=300px
                        # 容器高度设置，要等1.30版本更新，https://github.com/streamlit/streamlit/issues/2169
                        show_chat_message_from_history()
                        response(
                            prompt=input_text,
                            imgs=imgs,
                            autoplay=True,
                            audio_response=True
                        )
            else:
                st.toast("无效的语音输入，请重试。", icon="🤖")
                time.sleep(2)# 给2秒时间，调整准备输入
            print('对话完毕，释放录音锁，打开对话锁')
            # 对话响应完毕，打开事件
            ChatEngine.event_record.set()
            # 如果没有录入输入，等待
            ChatEngine.event_chat.clear()
            chat_placeholder.empty()


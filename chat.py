import streamlit as st
import random
import time
from openai import OpenAI
import uuid
from pathlib import Path
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from utils.audio2text import audio2text
from utils.audio_improve import process_audio
from utils.extracte_img import get_main_img
from utils.get_gpt4v_response import gpt4v
from utils.text2audio import text2audio

st.title("Gemini-like对话测试")
RECORD_DIR = Path("./records")
RECORD_DIR.mkdir(exist_ok=True)
# img={'assistant':'bot.png','user':None}
img={'assistant':None,'user':None}

def app():
    """
    该方法位录制视频和声音，单独提取声音
    """
    if "prefix" not in st.session_state:
        st.session_state["prefix"] = str(uuid.uuid4())
    prefix = st.session_state["prefix"]
    in_file_video = RECORD_DIR / f"{prefix}_input_video.mp4"
    in_file_audio = RECORD_DIR / f"{prefix}_input_audio.mp3"

    def in_recorder_factory_video() -> MediaRecorder:
        video_file = MediaRecorder(str(in_file_video), format="mp4")
        return video_file
    def in_recorder_factory_audio() -> MediaRecorder:
        audio_file = MediaRecorder(str(in_file_audio), format="mp3")
        return audio_file
    webrtc_streamer(
        key="record",
        mode=WebRtcMode.SENDRECV,
        media_stream_constraints={
            "video": True,
            "audio": True,
        },
        audio_frame_callback=process_audio,
        in_recorder_factory=in_recorder_factory_video,
        out_recorder_factory=in_recorder_factory_audio,
        async_processing=True
    )
    return in_file_video,in_file_audio
r_video_file, r_audio_file = 'None',Path('./source/test.m4a')

@st.cache_data(show_spinner=False)
def preocess_video(v_file,a_file):
    if a_file.exists():
        with st.status('处理录入的视频中...',state='running',expanded=True) as status:
            st.write('抽取关键帧...')
            imgs = get_main_img(v_file)
            cls = st.columns(len(imgs))
            for idx,cl in enumerate(cls):
                cl.image(imgs[idx])
            st.write('音频转文本...')
            st.info(f'缓存好的音频：{a_file}')
            st.audio(str(a_file))
            input_text_from_audio = audio2text(str(a_file))
            st.info('audio2text:')
            st.text(input_text_from_audio)
            status.update(label="录入视频文件处理完成", state="complete", expanded=False)
    return imgs,input_text_from_audio

# client = OpenAI(api_key="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")
# Set a default model
# if "openai_model" not in st.session_state:
#     st.session_state["openai_model"] = "gpt-3.5-turbo"

imgs,input_text_from_audio = preocess_video(r_video_file,r_audio_file)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
 
# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"],avatar=img[message['role']]):
        st.markdown(message["content"])
        if message['img'] is not None:
            st.image(message['img'])
        if message['audio'] is not None:
            st.audio(message['audio'])
 
# Accept user input
# if prompt := st.chat_input("What is up?"):
prompt = input_text_from_audio
if prompt:
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
 
    # Display assistant response in chat message container
    # with st.chat_message("assistant",avatar='bot.png'):
    with st.chat_message("assistant"):
        # res = client.chat.completions.create(
        #         model=st.session_state["openai_model"],
        #         messages=[{"role": m["role"], "content": m["content"]} for m in st.session_state.messages],
        #         stream=True
        #         )
        res = gpt4v(prompt,imgs)
        # message_placeholder = st.empty()
        # full_response = ""
        # for response in res:
        #     full_response += res
        #     message_placeholder.markdown(full_response + "▌")
        # message_placeholder.markdown(full_response)
        sound = text2audio(res["text"])
        st.markdown(res['text'])
        st.audio(sound)
        st.image(res['imgs'])
    st.session_state.messages.append({"role": "assistant", "content": res['text'],'imgs':res['imgs'],'audio':sound})
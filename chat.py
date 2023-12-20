import uuid
import streamlit as st
from pathlib import Path
from utils.audio2text import audio2text
from utils.extracte_img import get_main_img
from utils.get_gpt4v_response import gpt4v
from utils.text2audio import text2audio
from utils.record_video import record_video

st.title("Gemini-like对话测试")
RECORD_DIR = Path("./records")
RECORD_DIR.mkdir(exist_ok=True)
# 这里可能需要修改，不然会往一个路径里面记录
if "prefix" not in st.session_state:
    st.session_state["prefix"] = str(uuid.uuid4())
prefix = st.session_state["prefix"]
in_file_video = RECORD_DIR / f"{prefix}_input_video.mp4"
in_file_audio = RECORD_DIR / f"{prefix}_input_audio.mp3"
# 对话机器人的图标
img={'assistant':'./source/bot.png','user':None}

# 录制视频语音
r_video_file, r_audio_file = record_video(in_file_video,in_file_audio)

# @st.cache_data(show_spinner=False)
def preocess_video(v_file,a_file):
    imgs = None
    input_text_from_audio=None
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

imgs,input_text_from_audio = preocess_video(r_video_file,r_audio_file)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"],avatar=img[message['role']]):
        st.markdown(message["content"])
        try:
            if message['audio'] is not None:
                st.audio(message['audio'],sample_rate=24000)
        except:
            pass
        try:
            if message['img'] is not None:
                st.image(message['img'])
        except:
            pass
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
    with st.chat_message("assistant",avatar='./source/bot.png'):
        res = gpt4v(query=prompt,imgs=imgs)
        sound,rate = text2audio(res["text"])
        st.markdown(res['text'])
        st.audio(sound,sample_rate=rate)
        try:
            st.image(res['imgs'])
        except:
            pass
    st.session_state.messages.append({"role": "assistant", "content": res['text'],'audio':sound})
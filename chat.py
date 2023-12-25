import uuid
import streamlit as st
from pathlib import Path
from utils.audio2text import audio2text_from_bytes
from utils.extracte_img import get_main_img
from utils.get_gpt4v_response import gpt4v
from utils.text2audio import text2audio
from utils.record_video import record
import base64

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
# r_video_file = record_video(in_file_video)

# @st.cache_data(show_spinner=False)
# imgs,input_text_from_audio = preocess_video(r_video_file)
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []


def show_chat_message_from_history():
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
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
# Accept user input
# if prompt := st.chat_input("What is up?"):
def response(prompt=None,imgs=None):
    if prompt:
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display assistant response in chat message container
        with st.chat_message("assistant",avatar='./source/bot.png'):
            res = gpt4v(query=prompt,imgs=imgs)
            sound,rate,bytes_audio = text2audio(res["text"])
            # st.audio(sound,sample_rate=rate)
            autoplay_audio(bytes_audio)
            st.markdown(res['text'])
            try:
                st.image(res['imgs'])
            except:
                pass
        st.session_state.messages.append({"role": "assistant", "content": res['text'],'audio':sound})

def autoplay_audio(bytes_audio):
    # https://discuss.streamlit.io/t/how-to-play-an-audio-file-automatically-generated-using-text-to-speech-in-streamlit/33201/6
    b64 = base64.b64encode(bytes_audio).decode()
    md = f"""
        <audio controls autoplay="true">
        <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
        </audio>
        """
    st.markdown(
        md,
        unsafe_allow_html=True,
    )

if __name__ == '__main__':
    video_show = st.container()
    video_show.camera_input('tt',label_visibility='hidden')
    imgs,audio = None,None
    if video_show.button('开始对话'):
        imgs,audio = record()
    if audio is None:
        st.info('没有检测到输入')
        # st.stop()
        pass
    else:
        with st.status('处理输入信号...',state='running',expanded=True) as status:
            if len(imgs)>0:
                st.write('getMainFrame...')
                imgs = get_main_img(imgs)
                imgs = imgs[-3:]
                cls = st.columns(min(3,len(imgs)))
                for idx,cl in enumerate(cls):
                    cl.image(imgs[idx])
            st.write('audio2text...')
            st.audio(audio.get_wav_data())
            input_text,code_status,request_id = audio2text_from_bytes(audio.get_wav_data())
            # input_text = '图片里面有几个人'
            if input_text:
                st.text(f'识别后的文本：{input_text}')
            else:
                st.error(f'语音识别失败，code: {code_status} request_id: {request_id}')
            status.update(label="输入信号处理完成", state="complete", expanded=False)
        show_chat_message_from_history()
        response(prompt=input_text,imgs=imgs)
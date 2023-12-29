import streamlit as st
img={'assistant':'./source/bot.png','user':None}
if "messages" not in st.session_state:
    st.session_state.messages = []
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
        
show_chat_message_from_history()
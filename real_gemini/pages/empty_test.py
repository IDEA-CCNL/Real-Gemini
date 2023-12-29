import streamlit as st
import time

# with st.empty():
#     for seconds in range(10):
#         st.write(f"⏳ {seconds} seconds have passed")
#         time.sleep(1)
#     st.write("✔️ 1 minute over!")


placeholder = st.empty()

# Replace the placeholder with some text:
# placeholder.text("Hello")
# time.sleep(5)
# # Replace the text with a chart:
# placeholder.line_chart({"data": [1, 5, 2, 6]})
# st.text('other')
# time.sleep(5)
# # Replace the chart with several elements:
# with placeholder.container():
#     st.write("This is one element")
#     time.sleep(5)
#     st.write("This is another")
# time.sleep(5)
for i in range(10):
    with placeholder.status('doing',expanded=True,state='running') as status:
        st.text(f'这是测试{i}')
        time.sleep(2)
        status.update(label="done", state="complete", expanded=False)
    if i % 2 == 0:
        with st.chat_message('user'):
            st.text(f'user:test text {i}')
    else:
        with st.chat_message('assistant'):
            st.text(f'bot:test text {i}')
# Clear all those elements:
# placeholder.empty()
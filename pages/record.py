import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode

class VideoRecorder(VideoProcessorBase):
    def __init__(self):
        self.frames = []

    def recv(self, frame):
        print('see me!',frame)
        self.frames.append(frame)
        return frame

def main():
    st.title("High-Definition Video Recorder")

    webrtc_ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoRecorder,
        mode=WebRtcMode.SENDRECV,
        async_processing=True,
        media_stream_constraints={
            'audio':{},
            'video':{'height':360,'width':640}
        },
        video_receiver_size=10,
        audio_receiver_size=10,
    )

    if webrtc_ctx.video_processor:
        print(
            webrtc_ctx.video_processor.frames
        )
        print(len(webrtc_ctx.video_processor.frames))
        for img in webrtc_ctx.video_processor.frames:
            st.image(img)
        # st.video(webrtc_ctx.video_processor.frames)

    if st.button("Save Video"):
        if webrtc_ctx.video_processor:
            save_path = "recorded_video.mp4"
            webrtc_ctx.video_processor.frames.save_as_video(save_path)
            st.success(f"Video saved at {save_path}")

if __name__ == "__main__":
    main()

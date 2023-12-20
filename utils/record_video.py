import av
import pydub
import numpy as np
from aiortc.contrib.media import MediaRecorder
from streamlit_webrtc import WebRtcMode, webrtc_streamer
from streamlit_mic_recorder import mic_recorder,speech_to_text

def process_audio(frame: av.AudioFrame) -> av.AudioFrame:
    raw_samples = frame.to_ndarray()
    sound = pydub.AudioSegment(
        data=raw_samples.tobytes(),
        sample_width=frame.format.bytes,
        frame_rate=frame.sample_rate,
        channels=len(frame.layout.channels),
    )
    sound = sound.apply_gain(0)
    # Ref: https://github.com/jiaaro/pydub/blob/master/API.markdown#audiosegmentget_array_of_samples  # noqa
    channel_sounds = sound.split_to_mono()
    channel_samples = [s.get_array_of_samples() for s in channel_sounds]
    new_samples: np.ndarray = np.array(channel_samples).T
    new_samples = new_samples.reshape(raw_samples.shape)
    new_frame = av.AudioFrame.from_ndarray(new_samples, layout=frame.layout.name)
    new_frame.sample_rate = frame.sample_rate
    return new_frame

def record_video(video_save_path,audio_save_path):
    """
    该方法位录制视频和声音，单独提取声音
    """
    def in_recorder_factory_video() -> MediaRecorder:
        video_file = MediaRecorder(str(video_save_path), format="mp4")
        return video_file
    def in_recorder_factory_audio() -> MediaRecorder:
        audio_file = MediaRecorder(str(audio_save_path), format="mp3")
        return audio_file
    webrtc_streamer(
        key="video_record",
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
    return video_save_path,audio_save_path

def record_audio():
    audio=mic_recorder(
            start_prompt="开始录音",
            stop_prompt="录音结束", 
            just_once=False,
            use_container_width=False,
            callback=None,
            args=(),
            kwargs={},
            key='audio_record'
        )
    return audio
    
def audio2text(language):
    text=speech_to_text(
            language=language,
            start_prompt="Start recording",
            stop_prompt="Stop recording", 
            just_once=False,
            use_container_width=False,
            callback=None,
            args=(),
            kwargs={},
            key='speech2text'
        )
    print(text)
    return text

if __name__ == '__main__':
    import streamlit as st
    st.info('test')
    audio = record_audio()
    if audio:
        st.audio(audio['bytes'])
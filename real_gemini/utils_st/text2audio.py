import base64
import json
import io
import requests
import numpy as np
import streamlit as st
from numpy import typing as npt
from typing import Any,Tuple,Optional,cast,Union
from typing_extensions import Final, TypeAlias
MediaData: TypeAlias = Union[
    str, bytes, io.BytesIO, io.RawIOBase, io.BufferedReader, "npt.NDArray[Any]", None
]

def text2audio(text,):
    headers = {
        'accept': 'application/json',
        'Content-Type': 'application/x-www-form-urlencoded',
    }
    data = {
        'prompt': text,
    }
    response = requests.post('http://192.168.81.12:6679/tts/', headers=headers, data=data)
    res = response.json()
    audio_array = np.frombuffer(base64.b64decode(res[0]),np.float32)
    rate = res[1]
    return audio_array,rate,convert_to_wav_bytes(audio_array,rate)

def _validate_and_normalize(data: "npt.NDArray[Any]") -> Tuple[bytes, int]:
    """Validates and normalizes numpy array data.
    We validate numpy array shape (should be 1d or 2d)
    We normalize input data to int16 [-32768, 32767] range.

    Parameters
    ----------
    data : numpy array
        numpy array to be validated and normalized

    Returns
    -------
    Tuple of (bytes, int)
        (bytes, nchan)
        where
         - bytes : bytes of normalized numpy array converted to int16
         - nchan : number of channels for audio signal. 1 for mono, or 2 for stereo.
    """
    # we import numpy here locally to import it only when needed (when numpy array given
    # to st.audio data)
    import numpy as np

    data: "npt.NDArray[Any]" = np.array(data, dtype=float)

    if len(data.shape) == 1:
        nchan = 1
    elif len(data.shape) == 2:
        # In wave files,channels are interleaved. E.g.,
        # "L1R1L2R2..." for stereo. See
        # http://msdn.microsoft.com/en-us/library/windows/hardware/dn653308(v=vs.85).aspx
        # for channel ordering
        nchan = data.shape[0]
        data = data.T.ravel()
    else:
        raise "Numpy array audio input must be a 1D or 2D array."

    if data.size == 0:
        return data.astype(np.int16).tobytes(), nchan

    max_abs_value = np.max(np.abs(data))
    # 16-bit samples are stored as 2's-complement signed integers,
    # ranging from -32768 to 32767.
    # scaled_data is PCM 16 bit numpy array, that's why we multiply [-1, 1] float
    # values to 32_767 == 2 ** 15 - 1.
    np_array = (data / max_abs_value) * 32767
    scaled_data = np_array.astype(np.int16)
    return scaled_data.tobytes(), nchan

def _make_wav(data: "npt.NDArray[Any]", sample_rate: int) -> bytes:
    """
    Transform a numpy array to a PCM bytestring
    We use code from IPython display module to convert numpy array to wave bytes
    https://github.com/ipython/ipython/blob/1015c392f3d50cf4ff3e9f29beede8c1abfdcb2a/IPython/lib/display.py#L146
    """
    # we import wave here locally to import it only when needed (when numpy array given
    # to st.audio data)
    import wave

    scaled, nchan = _validate_and_normalize(data)

    with io.BytesIO() as fp, wave.open(fp, mode="wb") as waveobj:
        waveobj.setnchannels(nchan)
        waveobj.setframerate(sample_rate)
        waveobj.setsampwidth(2)
        waveobj.setcomptype("NONE", "NONE")
        waveobj.writeframes(scaled)
        return fp.getvalue()


def convert_to_wav_bytes(
    data: MediaData, sample_rate: Optional[int]
) -> MediaData:
    """Convert data to wav bytes if the data type is numpy array."""
    data = _make_wav(cast("npt.NDArray[Any]", data), sample_rate)
    return data

def autoplay_audio(bytes_audio):
    # https://discuss.streamlit.io/t/how-to-play-an-audio-file-automatically-generated-using-text-to-speech-in-streamlit/33201/6
    if isinstance(bytes_audio,str):
        with open(bytes_audio,'rb') as f:
            b64_audio = f.read()
    else:
        b64_audio = base64.b64encode(bytes_audio).decode()
    md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/wav;base64,{b64_audio}" type="audio/wav">
            </audio>
            """
    st.markdown(
        md,
        unsafe_allow_html=True,
    )

if __name__ == '__main__':
    with open('/Users/wuziwei/git_project/Real-Gemini/records/180367f8-85d3-4ec3-81dc-95e9c095b7ec_input_audio.mp3','rb') as f:
        ba = f.read()
        # print(ba)
    a,r,ba = text2audio('你好')
    autoplay_audio(ba)
    st.audio(a,sample_rate=r)
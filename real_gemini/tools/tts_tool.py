#encoding=utf8

import os
import json
import scipy
import base64
import hashlib
import requests
import torchaudio
import numpy as np
from ..utils_st.text2audio import convert_to_wav_bytes

class TTSTool(object):
    _name_ = "Text To Speech"
    _description_ = "这个工具是从文本转语音的调用接口，它可以根据一段文字，生成符合这段文本的wav语音。本工具的输入是一段文本指令。This tool is a text-to-speech API interface, which can generate a wav voice consistent with a piece of text based on it. The input of this tool is a piece of text command."
    _return_direct_ = True

    def __init__(self):
        self.host = os.getenv("TTS_SERVER_HOST")
        self.port = os.getenv("TTS_SERVER_PORT")
    
    def inference(self, input_str: str):

        url = f"http://{self.host}:{self.port}/tts"
        data = {"prompt": input_str}
        response = requests.post(url, data=data)
        response = response.json()

        audio_array = np.frombuffer(base64.b64decode(response["audio"]), np.float32)
        rate = response["sample_rate"]

        # write to file
        # save_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        # save_dir = os.path.join(save_dir, "test", "outputs")
        # md5 = hashlib.md5()
        # md5.update(input_str.encode('utf-8'))
        # filename = os.path.join(save_dir, md5.hexdigest() + ".wav")
        # torchaudio.save(filename, audio_array, rate)

        return audio_array, rate, convert_to_wav_bytes(audio_array, rate)

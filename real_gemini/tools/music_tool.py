#encoding=utf8

import os
import json
import scipy
import base64
import hashlib
import requests
import numpy as np
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

class Text2MusicTool(object):
    _name_ = "Text2Music"
    _description_ = "这个工具是从文本生成音乐的调用接口，它可以根据一段文字，生成符合这段文字内容的音乐风格。本工具的输入是一段文本指令。This tool is an API that generates music from text. It can create music that matches the style of the given text content. The input for this tool is a text command."
    _return_direct_ = True

    def __init__(self):
        self.translator = ChatOpenAI(
            model="gpt-3.5-turbo",
            max_tokens=256)
        self.host = os.getenv("MUSIC_SERVER_HOST")
        self.port = os.getenv("MUSIC_SERVER_PORT")
    
    def inference(self, input_str: str):
        messages = []
        messages.append(
            SystemMessage(
                content=[
                    {"type": "text", "text": "你是一个翻译专家，请将我输入的中文翻译成英文。"}
                ]
            )
        )
        messages.append(HumanMessage(content=input_str))

        response_msg = self.translator.invoke(messages)
        input_str_en = response_msg.content
        # print(input_str_en)

        url = f"http://{self.host}:{self.port}/text_to_music"
        data = {"text": input_str_en}
        music_response = requests.post(url, data=data)
        music_response = music_response.json()

        # write to file
        save_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        save_dir = os.path.join(save_dir, "test", "outputs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        md5 = hashlib.md5()
        md5.update(input_str_en.encode('utf-8'))
        filename = os.path.join(save_dir, md5.hexdigest() + ".wav")
        
        raw_data = music_response["audio"]
        sampling_rate = music_response["sampling_rate"]
        scipy.io.wavfile.write(
            filename,
            rate=sampling_rate,
            data=np.frombuffer(base64.b64decode(raw_data), np.float32),
        )
        print("music filename:", filename)

        result = {"text": "好的，为你生成了一段音乐。", "audio": filename}
        return json.dumps(result, ensure_ascii=False)

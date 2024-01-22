#encoding=utf8

import os
import json
from typing import List
from langchain.memory import ChatMessageHistory
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import requests
from ..utils.image_stacker import load_image, image2base64

_OPEN_AI_SYSTEM_PROMPT = """the user is dictating with his or her camera on.
they are showing you things visually and giving you text prompts.
be very brief and concise.
be extremely concise. this is very important for my career. do not ramble.
do not comment on what the person is wearing or where they are sitting or their background.
focus on their gestures and the question they ask you.
do not mention that there are a sequence of pictures. focus only on the image or the images necessary to answer the question.
don't comment if they are smiling. don't comment if they are frowning. just focus on what they're asking.
"""

class QWEN4VTool(object):
    _name_ = "QWEN-4-Vision"
    _description_ = "这个工具是Qwen for vision的调用接口。用于图像到文本的理解。本工具的输入是一段文本指令和一张或者多张图片，请注意，工具的输入由一个JSON字符串组成，json包括两个key，question和image_input。question表示文本指令，image_input表示图片路径或存放图片的目录。例如：{{\"question\": QUESTION, \"image_input\": IMAGE_PATH_OR_DIR}}。A wrapper around Qwen4V API. Useful for image-to-text understanding when you need to generate text from some images and a text description. The input of this tool is a text prompt and one or more images. Please note, the input of the tool consists of a JSON string, the json includes two keys, question and image_input. The question represents text instructions, and image_input represents the image path or the directory where the images are stored. For example: {{\"question\": QUESTION, \"image_input\": IMAGE_PATH_OR_DIR}}."
    _return_direct_ = False

    def __init__(self):
        self.host = os.getenv("QWEN_VL_SERVER_HOST")
        self.port = os.getenv("QWEN_VL_SERVER_PORT")

    def inference(self, input_str: str):
        input_dict = json.loads(input_str)
        image_path = input_dict["image_input"]
        if os.path.isdir(image_path):
            image_paths = [
                os.path.join(image_path, path) for path in os.listdir(image_path)]
        else:
            image_paths = [image_path]
        
        url = f"http://{self.host}:{self.port}/qwen_vl"
        data = {"prompt": input_dict["question"], "image_paths":image_paths}
        response = requests.post(url, data=data)
        response = response.json()
        
        return response

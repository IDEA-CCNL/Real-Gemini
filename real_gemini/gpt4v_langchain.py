#encoding=utf8

import os
from typing import List
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from .utils.image_stacker import load_image, image2base64

_OPEN_AI_SYSTEM_PROMPT = """the user is dictating with his or her camera on.
they are showing you things visually and giving you text prompts.
be very brief and concise.
be extremely concise. this is very important for my career. do not ramble.
do not comment on what the person is wearing or where they are sitting or their background.
focus on their gestures and the question they ask you.
do not mention that there are a sequence of pictures. focus only on the image or the images necessary to answer the question.
don't comment if they are smiling. don't comment if they are frowning. just focus on what they're asking.
"""

class GPT4VTool(object):
    _name_ = "GPT for vision"
    _description_ = "GPT for vision"

    def __init__(self, image_dir: str):
        self._image_dir = image_dir # 存放视频截帧的目录路径
        self._gpt4v = ChatOpenAI(
            model="gpt-4-vision-preview",
            max_tokens=256)
    
    def inference(self, prompt: str):
        image_paths = [
            os.path.join(self._image_dir, path) for path in os.listdir(self._image_dir)]
        base64_images = []
        for image_path in image_paths:
            base64_image = image2base64(load_image(image_path))
            base64_images.append(f"data:image/jpeg;base64,{base64_image}")

        messages = []
        messages.append(
            SystemMessage(
                content=[
                    {"type": "text", "text": _OPEN_AI_SYSTEM_PROMPT}
                ]
            )
        )
        human_contents = []
        human_contents.append({"type": "text", "text": prompt})
        for base64_image in base64_images:
            human_contents.append({
                "type": "image_url",
                "image_url": {"url": base64_image}
            }) # images
        messages.append(HumanMessage(content=human_contents))

        response_msg = self._gpt4v.invoke(messages)
        # print(response_msg.content)
        return response_msg.content

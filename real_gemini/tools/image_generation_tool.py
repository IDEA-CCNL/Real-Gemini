#encoding=utf8

import os
import json
import hashlib
import requests
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from ..utils.image_stacker import save_or_show_image

class TaiyiGeneralTool(object):
    _name_ = "taiyi general image generation"
    _description_ = "Taiyi General的API，用于从文本生成图像。当你需要从文本描述生成图像时非常有用。输入应该是文本，即图像描述。A wrapper around Taiyi General API for text to image generation. Useful for when you need to generate images from a text description. Input should be text, i.e, an image description."
    _return_direct_ = True

    def __init__(self):
        self.prompter = ChatOpenAI(
            model="gpt-3.5-turbo",
            max_tokens=256)
        self.host = os.getenv("IMAGE_GENERATION_SERVER_HOST")
        self.port = os.getenv("IMAGE_GENERATION_SERVER_PORT")
    
    def _upgrade_prompt(self, prompt):
        messages = []
        messages.append(
            SystemMessage(
                content=[
                    {"type": "text", "text": "我正在使用一个Stable Diffusion的AI图像生成工具，我想让你充当我的prompt优化生成器。在我想生成的主题后，请帮我添加各种关键词，使得我的主题的描述更加详细，添加的关键词包括：主体、背景效果、风格、拍摄方式。例如，如果我输入“跑车”，你将生成关键词，如：“跑车,高清,4k,真实细致的跑车摄影,速度动态模糊,赛车场,城市环境,风景道路,戏剧性的天空”"}
                ]
            )
        )
        messages.append(HumanMessage(content=prompt))

        response_msg = self.prompter.invoke(messages)
        new_prompt = response_msg.content
        return new_prompt

    def inference(self, inputs):
        url = f"http://{self.host}:{self.port}/taiyi_xl_general_base64/"
        headers = {"Content-Type": "application/json"}
        new_prompt = self._upgrade_prompt(inputs)
        print("new prompt:", new_prompt)
        data = {"prompt": new_prompt}
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response = response.json()
        b64_image = response["image_base64"]
        
        # write to file
        save_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        save_dir = os.path.join(save_dir, "test", "outputs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        md5 = hashlib.md5()
        md5.update(inputs.encode('utf-8'))
        filename = os.path.join(save_dir, md5.hexdigest() + ".png")
        save_or_show_image(b64_image, filename)
        
        print("image filename:", filename)

        result = {"text": "好的，我用太乙为你生成了一张图片。", "image": filename}
        return json.dumps(result, ensure_ascii=False)
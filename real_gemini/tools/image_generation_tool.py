#encoding=utf8

import os
import json
import hashlib
import requests
from ..utils.image_stacker import save_or_show_image

class TaiyiGeneralTool(object):
    _name_ = "taiyi general image generation"
    _description_ = "Taiyi General的API，用于从文本生成图像。当你需要从文本描述生成图像时非常有用。输入应该是文本，即图像描述。\nA wrapper around Taiyi General API for text to image generation. Useful for when you need to generate images from a text description. Input should be text, i.e, an image description."
    _return_direct_ = True

    def __init__(self):
        self.host = os.getenv("IMAGE_GENERATION_SERVER_HOST")
        self.port = os.getenv("IMAGE_GENERATION_SERVER_PORT")
    
    def inference(self, inputs):
        url = f"http://{self.host}:{self.port}/taiyi_xl_general_base64/"
        headers = {"Content-Type": "application/json"}
        data = {"prompt": inputs}
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
import base64
import json
import requests
from image_selector import ImageSelector

def get_main_img(imgs):
    # resp = requests.post(
    #     'http://192.168.80.29:8789/asr', 
    #     data=input_data, 
    #     headers={"Content-Type": "application/json"}
    #     )
    # resp_data = resp.json()
    # prompt_text = resp_data["text"]
    image_selector = ImageSelector(5)

    top_frames = image_selector.select_best_frames(
        imgs, 5
    )
    return top_frames
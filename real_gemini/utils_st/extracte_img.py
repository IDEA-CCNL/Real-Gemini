import base64
import json
import requests
import numpy as np
from utils_st.image_selector import ImageSelector

def get_main_img(imgs, num_frames):
    # resp = requests.post(
    #     'http://192.168.80.29:8789/asr', 
    #     data=input_data, 
    #     headers={"Content-Type": "application/json"}
    #     )
    # resp_data = resp.json()
    # prompt_text = resp_data["text"]
    image_selector = ImageSelector(5)

    top_frames = image_selector.select_best_frames(
        imgs, num_frames
    )

    sorted_frames = []
    for i in range(len(imgs)):
        for j in range(len(top_frames)):
            if np.all(imgs[i]==top_frames[j]):
                sorted_frames.append(imgs[i])

    return sorted_frames
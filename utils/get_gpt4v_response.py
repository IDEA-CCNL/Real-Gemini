import base64
import json
import requests

def gpt4v(query,imgs=None):
    import time
    time.sleep(2)
    # resp = requests.post(
    #     'http://192.168.80.29:8789/asr', 
    #     data=input_data, 
    #     headers={"Content-Type": "application/json"}
    #     )
    # resp_data = resp.json()
    # prompt_text = resp_data["text"]
    send = {
        'text':'这是一个测试！',
        'imgs':'./source/1702803359034.png'
    }
    return send
import base64
import json
import requests
import numpy as np

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
    return audio_array,rate


if __name__ == '__main__':
    a,r = text2audio('你好')
    print(a)
    print(r)
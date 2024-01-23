import base64
import json
import requests

def audio2text(fin):
    f = open(fin,'rb')
    audio_b64 = base64.b64encode(f.read()).decode()
    input_data = json.dumps({
        "audio_b64": audio_b64,
        'input_f': '',
        'input_ar': '',
        'input_ac': '',
        'input_acodec': '',
        })
    resp = requests.post(
        'http://192.168.80.29:8789/asr', 
        data=input_data, 
        headers={"Content-Type": "application/json"}
        )
    resp_data = resp.json()
    print(resp_data)
    prompt_text = resp_data["text"]
    f.close()
    return prompt_text


def audio2text_from_bytes(bytes_input):
    audio_b64 = base64.b64encode(bytes_input).decode()
    input_data = json.dumps({
        "audio_b64": audio_b64,
        'input_f': '',
        'input_ar': '',
        'input_ac': '',
        'input_acodec': '',
        'no_speech_chtreshold':0.75,
        'debug':True
        })
    resp = requests.post(
        'http://192.168.80.29:8789/asr', 
        data=input_data, 
        headers={"Content-Type": "application/json"}
        )
    resp_data = resp.json()
    prompt_text = resp_data["text"]
    code = resp_data['code']
    request_id = resp_data['request_id']
    return prompt_text,code,request_id

def audio2text_test(fin):
    import time 
    time.sleep(2)
    return 'audio2text 测试结果'

if __name__ == '__main__':
    r = audio2text('/Users/wuziwei/git_project/Real-Gemini/records/180367f8-85d3-4ec3-81dc-95e9c095b7ec_input_audio.mp3')
    print(r)
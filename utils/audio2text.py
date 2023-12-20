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
    prompt_text = resp_data["text"]
    f.close()
    return prompt_text

def audio2text_test(fin):
    import time 
    time.sleep(2)
    return 'audio2text 测试结果'

if __name__ == '__main__':
    r = audio2text('records/6f64cd30-bdb6-4705-b2fb-e9207149825b_input_audio.mp3')
    print(r)
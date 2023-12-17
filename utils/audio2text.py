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
    return prompt_text
import base64
import json
import requests

URL='http://192.168.77.1:8000/main/'
def img2base64(imgs):
    base64_imgs = []
    if isinstance(imgs,list):
        for img in imgs:
            with open(img,'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
                base64_imgs.append(img_b64)
        return base64_imgs
    else:
        with open(imgs,'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
            base64_imgs.append(img_b64)
        return base64_imgs

def gpt4v(query,imgs=None):
    imgs = img2base64(imgs)
    input_data = {
        'query':query,
        'base64_images':imgs
    }
    api_url = 'http://192.168.77.1:8000/main/'
    resp = requests.post(
        api_url, 
        headers={
            'Content-Type':'application/x-www-form-urlencoded',
            'accept':'application/json'
        },
        data=input_data, 
        )
    resp_data = resp.json()
    prompt_text = resp_data["response"]
    send = {
        'text':prompt_text
    }
    return send

if __name__ == '__main__':
    r = gpt4v(query='第二张图和第一张图片有什么内容？',imgs=['../source/1702803312680.png','../source/1702803359034.png'])
    print(r)
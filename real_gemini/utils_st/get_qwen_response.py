import base64
import requests
import numpy as np
import cv2
# from real_gemini.gpt4v import GPT4V


def img2base64(imgs):
    base64_imgs = []
    if isinstance(imgs,list):
        for img in imgs:
            # with open(img,'rb') as f:
            if isinstance(img,str):
                with open(img,'rb') as f:
                    img_bytes = f.read()
            else:
                img_bytes = np.array(cv2.imencode('.png', img)[1]).tobytes()
            img_b64 = base64.b64encode(img_bytes).decode()
            # print(img_b64)
            base64_imgs.append(img_b64)
        return base64_imgs
    else:
        # with open(imgs,'rb') as f:
        if isinstance(imgs,str):
            with open(imgs,'rb') as f:
                img_bytes = f.read()
        else:
            img_bytes = np.array(cv2.imencode('.png', img)[1]).tobytes()
        img_bytes = np.array(cv2.imencode('.png', img)[1]).tobytes()
        img_b64 = base64.b64encode(img_bytes).decode()
        base64_imgs.append(img_b64)
        return base64_imgs

def QwenVL_client(query,imgs=None):
    imgs = img2base64(imgs)
    input_data = {
        'prompt':query,
        'image_strs':imgs
    }
    api_url = 'http://192.168.80.19:6679/qwen-vl/'
    try:
        resp = requests.post(
            api_url, 
            headers={
                'Content-Type':'application/x-www-form-urlencoded',
                'accept':'application/json'
            },
            data=input_data, 
            )
        resp_data = resp.json()
        # print(resp_data)
        prompt_text = resp_data
    except Exception as e:
        print(e)
        prompt_text = '千问接口请求出错了，请确认后台服务后再尝试～'
    send = {
        'text':prompt_text
    }
    return send


if __name__ == '__main__':
    imgs = ['/Users/wuziwei/git_project/Real-Gemini/source/bot.png']
    r = QwenVL_client(query='描述一下这种图片',imgs=imgs)
    print(r)
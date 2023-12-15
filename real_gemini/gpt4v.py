# https://platform.openai.com/docs/guides/vision

from typing import List
import requests

from real_gemini.utils.image_stacker import scale_and_stack_images, load_images, save_or_show_image
from openai import OpenAI

OPEN_AI_SYSTEM_PROMPT = """the user is dictating with his or her camera on.
they are showing you things visually and giving you text prompts.
be very brief and concise.
be extremely concise. this is very important for my career. do not ramble.
do not comment on what the person is wearing or where they are sitting or their background.
focus on their gestures and the question they ask you.
do not mention that there are a sequence of pictures. focus only on the image or the images necessary to answer the question.
don't comment if they are smiling. don't comment if they are frowning. just focus on what they're asking.
"""
OPENAI_API_KEY = "YOUR_OEPNAI_API_KEY"

class GPT4V():
    def __init__(self) -> None:
        pass

    def process_request(
        self,
        query: str,
        image_paths: List[str]
    ):
        images = load_images(image_paths)
        stacked_image_base64 = scale_and_stack_images(images)
        # 保存或展示处理后的图像
        save_or_show_image(stacked_image_base64, "./test/stacked_image.jpg") # 保存到文件

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {OPENAI_API_KEY}"
            }

        payload = {
            "model": "gpt-4-vision-preview",
            "messages": [
                {
                "role": "user",
                "content": [
                    {
                    "type": "text",
                    "text": query,
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{stacked_image_base64}"
                    }
                    }
                ]
                }
            ],
            "max_tokens": 300
            }

        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

        # print(response.json())
        return response.json()

if __name__=="__main__":
    query = "Guess what movie I'm acting out."
    image_paths = ["./test/test_0.png", "./test/test_1.png", "./test/test_2.png", "./test/test_3.png"]
    print(query, image_paths)

    gpt4v = GPT4V()
    result = gpt4v.process_request(query=query, image_paths=image_paths)

    print(result)
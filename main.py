from fastapi import FastAPI, File, Form, UploadFile
from typing import List
import requests
from PIL import Image
from io import BytesIO
import base64
import json


from real_gemini.gpt4v import GPT4V

gpt4v = GPT4V()
app = FastAPI()


# API端点
@app.post("/main/")
async def main(
    query: str = Form(...),
    base64_images: List[str] = Form(...)
):  
    base64_images = base64_images[0].split(",")
    # print("len_base64_images",len(base64_images))
    result = gpt4v.process_multi_image_base64(query=query, base64_images=base64_images)

    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

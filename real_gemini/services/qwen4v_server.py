#encoding=utf8

import os
import sys
import shutil
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from real_gemini.utils.image_stacker import save_or_show_image

app = FastAPI()

global tokenizer
global model
tokenizer, model = None, None

def save_buf_image(image_buf_dir, base64_images):
    if os.path.exists(image_buf_dir):
        shutil.rmtree(image_buf_dir)
    os.makedirs(image_buf_dir)
    for i, base64_image in enumerate(base64_images):
        save_or_show_image(base64_image, os.path.join(image_buf_dir, f"tmp_image_{i}.jpg"))

# API端点
@app.post("/qwen_vl/")
async def qwen_vl(
    prompt: str = Form(...),
    base64_images: List[str] = Form(...)
):
    image_buf_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "qwen_img_buf")
    save_buf_image(image_buf_dir, base64_images)

    input_list = []
    for file in sorted(os.listdir(image_buf_dir)):
        image_path = os.path.join(image_buf_dir, file)
        input_list.append({'image': image_path})
    input_list.append({'text': prompt})

    query = tokenizer.from_list_format(input_list)

    response, history = model.chat(tokenizer, query=query, history=None)

    return response


if __name__ == "__main__":
    work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_file = os.path.join(work_dir, ".env")
    load_dotenv(dotenv_path=env_file)
    
    model_path=os.getenv("QWEN_VL_MODEL_PATH")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True, bf16=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

    uvicorn.run(
        app,
        host=os.getenv("QWEN_VL_SERVER_HOST"),
        port=int(os.getenv("QWEN_VL_SERVER_PORT"))
    )
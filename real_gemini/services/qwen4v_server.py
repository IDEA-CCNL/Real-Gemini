#encoding=utf8

import os
import io
import sys
import base64
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form
from typing import List
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

app = FastAPI()

global tokenizer
global model
tokenizer, model = None, None

# API端点
@app.post("/qwen_vl/")
async def qwen_vl(
    prompt: str = Form(...),
    image_paths: List[str] = Form(...)
):
    
    input_list = []
    for image_path in image_paths:
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
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda", trust_remote_code=True, bf16=True).eval()
    model.generation_config = GenerationConfig.from_pretrained(model_path, trust_remote_code=True)

    uvicorn.run(
        app,
        host=os.getenv("QWEN_VL_SERVER_HOST"),
        port=int(os.getenv("QWEN_VL_SERVER_PORT"))
    )
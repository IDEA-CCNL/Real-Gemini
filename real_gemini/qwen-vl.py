from fastapi import FastAPI, File, Form, UploadFile
from typing import List
app = FastAPI()

output_image_dir = "/cognitive_comp/common_checkpoint/output_images"
import torch, torchaudio
import uuid
import base64
import os
import random

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("/cognitive_comp/lujunyu/checkpoints/Qwen-VL-Chat", trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained("/cognitive_comp/lujunyu/checkpoints/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("/cognitive_comp/lujunyu/checkpoints/Qwen-VL-Chat", trust_remote_code=True)

# API端点
@app.post("/qwen-vl/")
async def qwen_vl(
    prompt: str = Form(...),
    image_strs: List[str] = Form(...)
):
    image_strs = image_strs[0].split(",")
    tmp_dir = str(random.randint(0, 10e9))
    os.mkdir(os.path.join(output_image_dir, tmp_dir))

    for i in range(len(image_strs)):
        imgdata = base64.b64decode(image_strs[i])
        #设置文件存储路径
        file = open(os.path.join(output_image_dir, tmp_dir, f"tmp_image_{i}.jpg"),'wb')
        file.write(imgdata)
        file.close()

    input_list = []
    for file in sorted(os.listdir(os.path.join(output_image_dir, tmp_dir))):
        image_path = os.path.join(output_image_dir, tmp_dir, file)
        input_list.append({'image': image_path})
    input_list.append({'text': prompt})

    query = tokenizer.from_list_format(input_list)

    response, history = model.chat(tokenizer, query=query, history=None)

    return response

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6678)

#encoding=utf8

import os
import io
import sys
import argparse
import uvicorn
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import Response, StreamingResponse

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from real_gemini.models.dino import Text2Box

app = FastAPI()

global text2box_model
text2box_model = None

@app.post("/text_to_box")
async def text2box(inputs: str = Form(...)):
    output = text2box_model(inputs)
    return {"box_image_path": output}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="text2box API")
    parser.add_argument("--host", type=str, default="localhost", help="Host")
    parser.add_argument("--port", type=int, default=6681, help="Port")
    # parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    text2box_model = Text2Box(device="cuda")

    uvicorn.run(app, host=args.host, port=args.port)
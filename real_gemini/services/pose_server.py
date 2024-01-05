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

from real_gemini.models.controlnet import Image2Pose

app = FastAPI()

global image2pose_model
image2pose_model = None

@app.post("/image_to_pose")
async def image_to_pose(image_input: str = Form(...)):
    output = image2pose_model(image_input)
    return {"pose_image_path": output}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="image2pose API")
    parser.add_argument("--host", type=str, default="localhost", help="Host")
    parser.add_argument("--port", type=int, default=6679, help="Port")
    # parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    image2pose_model = Image2Pose(device="cuda")

    uvicorn.run(app, host=args.host, port=args.port)
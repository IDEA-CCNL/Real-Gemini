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

from real_gemini.models.imageediting import ImageEditing, Inpainting
from real_gemini.models.dino import Text2Box
from real_gemini.models.sam import Segmenting

app = FastAPI()

global editing_model
editing_model = None

@app.post("/image_remove")
async def segmenting(input: str = Form(...)):
    output = editing_model.inference_remove(input)
    return {"updated_image_paths": output}

@app.post("/image_replace")
async def segmenting(input: str = Form(...)):
    output = editing_model.inference_replace_sam(input)
    return {"updated_image_paths": output}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="imageEditing API")
    parser.add_argument("--host", type=str, default="localhost", help="Host")
    parser.add_argument("--port", type=int, default=6682, help="Port")
    # parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    text2box_model = Text2Box(device="cuda")
    segment_model = Segmenting(device="cuda")
    inpaint_model = Inpainting(device="cuda")
    editing_model = ImageEditing(text2box_model, segment_model, inpaint_model)

    uvicorn.run(app, host=args.host, port=args.port)
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

from real_gemini.models.sam import Segmenting

app = FastAPI()

global segment_model
segment_model = None

@app.post("/segmenting")
async def segmenting(image_input: str = Form(...)):
    output = segment_model(image_input)
    return {"seg_image_path": output}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="segmenting API")
    parser.add_argument("--host", type=str, default="localhost", help="Host")
    parser.add_argument("--port", type=int, default=6680, help="Port")
    # parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    segment_model = Segmenting(device="cuda")

    uvicorn.run(app, host=args.host, port=args.port)
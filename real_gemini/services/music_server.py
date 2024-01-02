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

from real_gemini.models.text2music import Text2MusicModel

app = FastAPI()

def AudioResponse(audio):
    if isinstance(audio, (str, Path)):
        audio = open(audio, "rb")
    return StreamingResponse(audio, media_type="audio/wav")

global text2music_model
text2music_model = None

@app.post("/text_to_music")
async def text_to_music(text: str = Form(...)):
    output = text2music_model(text)
    # return AudioResponse(output)
    return {"wav_file_path": output}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audio API")
    parser.add_argument("--host", type=str, default="localhost", help="Host")
    parser.add_argument("--port", type=int, default=6678, help="Port")
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    text2music_model = Text2MusicModel(model_path=args.model_path, device="cuda")

    uvicorn.run(app, host=args.host, port=args.port)
#encoding=utf8

import os
import io
import sys
import uvicorn
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Form
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
    return {"audio": output}

if __name__ == "__main__":
    work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_file = os.path.join(work_dir, ".env")
    load_dotenv(dotenv_path=env_file)

    
    text2music_model = Text2MusicModel(model_path=os.getenv("MUSIC_MODEL_PATH"), device="cuda")

    uvicorn.run(
        app,
        host=os.getenv("MUSIC_SERVER_HOST"),
        port=int(os.getenv("MUSIC_SERVER_PORT"))
    )
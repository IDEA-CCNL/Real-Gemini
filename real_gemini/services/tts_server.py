#encoding=utf8

import os
import io
import sys
import base64
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Form

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from real_gemini.models.tts import TTSModel

app = FastAPI()

global tts_model
tts_model = None

# API端点
@app.post("/tts/")
async def tts(prompt: str = Form(...)):
    wav, sample_rate = tts_model(prompt)
    wav = base64.b64encode(wav.tobytes())
    return {"audio": wav, "sample_rate": sample_rate}

if __name__ == "__main__":
    work_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    env_file = os.path.join(work_dir, ".env")
    load_dotenv(dotenv_path=env_file)

    tts_model = TTSModel(
        model_path=os.getenv("TTS_MODEL_PATH"),
        ref_wav_path=os.getenv("REFERENCE_WAV_PATH"),
        device="cuda")

    uvicorn.run(
        app,
        host=os.getenv("TTS_SERVER_HOST"),
        port=int(os.getenv("TTS_SERVER_PORT"))
    )
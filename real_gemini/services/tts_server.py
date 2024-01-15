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
    # print(out["wav"].numpy().dtype)
    # # print(ret_wav)
    # import numpy
    # wav = numpy.frombuffer(base64.b64decode(ret_wav), dtype=numpy.float32)
    # wav = torch.tensor(wav).unsqueeze(dim=0)
    # print("wav:", wav)
    # # Generate a random UUID for use as a filename
    # random_uuid_for_filename = uuid.uuid4()
    # file_path=f"{output_wav_dir}/{random_uuid_for_filename}.wav"
    # torchaudio.save(file_path, wav, 24000)
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
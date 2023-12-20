from fastapi import FastAPI, File, Form, UploadFile
from typing import List
app = FastAPI()

xtts_path = "/cognitive_comp/common_checkpoint/XTTS-v2"
ref_wav_path = "/cognitive_comp/common_checkpoint/xtts_ref_audio/female.wav"
output_wav_dir = "/cognitive_comp/common_checkpoint/output_xtts_audio"
import torch, torchaudio
import uuid
import base64

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Init TTS
# prompt = "吃葡萄不吐葡萄皮儿，不吃葡萄倒吐葡萄皮儿。"
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

config = XttsConfig()
config.load_json(f"{xtts_path}/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=xtts_path, eval=True)
model.cuda()

# API端点
@app.post("/tts/")
async def tts(
    prompt: str = Form(...)
):
    print(prompt)
    out = model.synthesize(
        prompt,
        config,
        speaker_wav=ref_wav_path,
        gpt_cond_len=3,
        language="zh-cn",
    )
    out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
    ret_wav = base64.b64encode(out["wav"].numpy().tobytes())
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
    return ret_wav, 24000

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=6679)

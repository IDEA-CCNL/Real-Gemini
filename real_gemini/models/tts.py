#encoding=utf8

import os
import torch
import langid
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

class TTSModel(object):
    def __init__(self, model_path: str, ref_wav_path: str, device="cuda"):
        self.device = device
        self.dtype = torch.float16
        # init model
        config = XttsConfig()
        config.load_json(os.path.join(model_path, "config.json"))
        self.model = Xtts.init_from_config(config)
        self.model.load_checkpoint(config, checkpoint_dir=model_path, eval=True)
        self.model.to(device=self.device)
        (self.gpt_cond_latent, self.speaker_embedding,) = self.model.get_conditioning_latents(
                audio_path=ref_wav_path, gpt_cond_len=30, gpt_cond_chunk_len=4, max_ref_length=60)
        self.sample_rate = 24000

    def __call__(self, text: str):
        language_predicted = langid.classify(text)[0].strip()  # strip need as there is space at end!

        # tts expects chinese as zh-cn
        if language_predicted == "zh":
            # we use zh-cn
            language_predicted = "zh-cn"
        print(f"Detected language: {language_predicted}")
        out = self.model.inference(
            text,
            language_predicted,
            self.gpt_cond_latent,
            self.speaker_embedding,
            repetition_penalty=5.0,
            temperature=0.75,
        )
        out["wav"] = torch.tensor(out["wav"]).unsqueeze(0)
        return out["wav"].numpy(), self.sample_rate
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

    def to(self, device):
        self.device = device
        self.model.to(device)
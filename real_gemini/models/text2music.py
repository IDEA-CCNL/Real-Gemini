#encoding=utf8

import os
import torch
import scipy
import base64
import hashlib
import numpy as np
from transformers import (
    AutoProcessor,
    MusicgenForConditionalGeneration,
)

class Text2MusicModel(object):
    def __init__(self, model_path: str, device="cuda"):
        self.device = device
        self.dtype = torch.float16
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = MusicgenForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=self.dtype
        )
        self.model.to(device=self.device)

    def __call__(self, text: str):
        inputs = self.processor(
            text=[text],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        audio_values = self.model.generate(**inputs, max_new_tokens=512)
        
        data = audio_values[0, 0].cpu().numpy().astype(np.float32)
        sampling_rate = self.model.config.audio_encoder.sampling_rate
        raw_data = base64.b64encode(data.tobytes())
        return raw_data, sampling_rate

    def to(self, device):
        self.device = device
        self.model.to(device)

if __name__ == "__main__":
    model = Text2MusicModel('cuda')
    print(
        model(
            "An 80s driving pop song with heavy drums and synth pads in the background"
        )
    )
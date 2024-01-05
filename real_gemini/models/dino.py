import os
from PIL import Image, ImageDraw, ImageOps, ImageFont
import wget
import numpy as np
import torch
import uuid

import groundingdino.datasets.transforms as T
from groundingdino.models import build_model
from groundingdino.util import box_ops
from groundingdino.util.slconfig import SLConfig
from groundingdino.util.utils import clean_state_dict, get_phrases_from_posmap

from .utils import get_new_image_name


class Text2Box(object):
    def __init__(self, device="cuda"):
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32
        self.model_checkpoint_path = os.path.join("checkpoints","groundingdino")
        self.model_config_path = os.path.join("checkpoints","grounding_config.py")
        self.download_parameters()
        self.box_threshold = 0.3
        self.text_threshold = 0.25
        self.grounding = (self.load_model()).to(self.device)

    def download_parameters(self):
        url = "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth"
        if not os.path.exists(self.model_checkpoint_path):
            wget.download(url,out=self.model_checkpoint_path)
        config_url = "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/groundingdino/config/GroundingDINO_SwinT_OGC.py"
        if not os.path.exists(self.model_config_path):
            wget.download(config_url,out=self.model_config_path)
    def load_image(self,image_path):
         # load image
        image_pil = Image.open(image_path).convert("RGB")  # load image

        transform = T.Compose(
            [
                T.RandomResize([512], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image_pil, image

    def load_model(self):
        args = SLConfig.fromfile(self.model_config_path)
        args.device = self.device
        model = build_model(args)
        checkpoint = torch.load(self.model_checkpoint_path, map_location="cpu")
        load_res = model.load_state_dict(clean_state_dict(checkpoint["model"]), strict=False)
        print(load_res)
        _ = model.eval()
        return model

    def get_grounding_boxes(self, image, caption, with_logits=True):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        image = image.to(self.device)
        with torch.no_grad():
            outputs = self.grounding(image[None], captions=[caption])
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        logits.shape[0]

        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        logits_filt.shape[0]

        # get phrase
        tokenlizer = self.grounding.tokenizer
        tokenized = tokenlizer(caption)
        # build pred
        pred_phrases = []
        for logit, box in zip(logits_filt, boxes_filt):
            pred_phrase = get_phrases_from_posmap(logit > self.text_threshold, tokenized, tokenlizer)
            if with_logits:
                pred_phrases.append(pred_phrase + f"({str(logit.max().item())[:4]})")
            else:
                pred_phrases.append(pred_phrase)

        return boxes_filt, pred_phrases
    
    def plot_boxes_to_image(self, image_pil, tgt):
        H, W = tgt["size"]
        boxes = tgt["boxes"]
        labels = tgt["labels"]
        assert len(boxes) == len(labels), "boxes and labels must have same length"

        draw = ImageDraw.Draw(image_pil)
        mask = Image.new("L", image_pil.size, 0)
        mask_draw = ImageDraw.Draw(mask)

        # draw boxes and masks
        for box, label in zip(boxes, labels):
            # from 0..1 to 0..W, 0..H
            box = box * torch.Tensor([W, H, W, H])
            # from xywh to xyxy
            box[:2] -= box[2:] / 2
            box[2:] += box[:2]
            # random color
            color = tuple(np.random.randint(0, 255, size=3).tolist())
            # draw
            x0, y0, x1, y1 = box
            x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)

            draw.rectangle([x0, y0, x1, y1], outline=color, width=6)
            # draw.text((x0, y0), str(label), fill=color)

            font = ImageFont.load_default()
            if hasattr(font, "getbbox"):
                bbox = draw.textbbox((x0, y0), str(label), font)
            else:
                w, h = draw.textsize(str(label), font)
                bbox = (x0, y0, w + x0, y0 + h)
            # bbox = draw.textbbox((x0, y0), str(label))
            draw.rectangle(bbox, fill=color)
            draw.text((x0, y0), str(label), fill="white")

            mask_draw.rectangle([x0, y0, x1, y1], fill=255, width=2)

        return image_pil, mask
    
    def __call__(self, inputs):
        try:
            image_input, det_prompt = inputs.split(",")
        except:
            return "Please input the image_dir and the object to be detected, a comma seperated string of two."
        print(f"image_input={image_input}, text_prompt={det_prompt}")
        # 如果是目录，就遍历目录下的所有图片，不然就遍历输入中的所有图片
        if os.path.isdir(image_input):
            image_paths = [
                os.path.join(image_input, path) for path in os.listdir(image_input)
            ]
        else:
            image_paths = image_input.split(",")
        updated_image_paths = []
        for image_path in image_paths:
            if "detect-something" in image_path:
                continue
            image_pil, image = self.load_image(image_path)

            boxes_filt, pred_phrases = self.get_grounding_boxes(image, det_prompt)

            size = image_pil.size
            pred_dict = {
            "boxes": boxes_filt,
            "size": [size[1], size[0]],  # H,W
            "labels": pred_phrases,}

            image_with_box = self.plot_boxes_to_image(image_pil, pred_dict)[0]

            updated_image_path = get_new_image_name(image_path, func_name="detect-something")
            updated_image_paths.append(updated_image_path)
            updated_image = image_with_box.resize(size)
            updated_image.save(updated_image_path)
            print(
                f"\nProcessed ObejectDetecting, Input Image: {image_path}, Object to be Detect {det_prompt}, "
                f"Output Image: {updated_image_path}")
        return updated_image_paths
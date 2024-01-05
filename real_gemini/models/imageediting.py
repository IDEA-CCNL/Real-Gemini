import os
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw, ImageFont

from diffusers import StableDiffusionInpaintPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker

from .dino import Text2Box
from .sam import Segmenting

from .utils import get_new_image_name

class Inpainting:
    def __init__(self, device):
        self.device = device
        self.revision = 'fp16' if 'cuda' in self.device else None
        self.torch_dtype = torch.float16 if 'cuda' in self.device else torch.float32

        self.inpaint = StableDiffusionInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting", revision=self.revision, torch_dtype=self.torch_dtype,safety_checker=StableDiffusionSafetyChecker.from_pretrained('CompVis/stable-diffusion-safety-checker')).to(device)
    def __call__(self, prompt, image, mask_image, height=512, width=512, num_inference_steps=50):
        update_image = self.inpaint(prompt=prompt, image=image.resize((width, height)),
                                     mask_image=mask_image.resize((width, height)), height=height, width=width, num_inference_steps=num_inference_steps).images[0]
        return update_image

class ImageEditing(object):
    template_model = True
    def __init__(self, Text2Box:Text2Box, Segmenting:Segmenting, Inpainting:Inpainting):
        self.sam = Segmenting
        self.grounding = Text2Box
        self.inpaint = Inpainting

    def pad_edge(self,mask,padding):
        #mask Tensor [H,W]
        mask = mask.numpy()
        true_indices = np.argwhere(mask)
        mask_array = np.zeros_like(mask, dtype=bool)
        for idx in true_indices:
            padded_slice = tuple(slice(max(0, i - padding), i + padding + 1) for i in idx)
            mask_array[padded_slice] = True
        new_mask = (mask_array * 255).astype(np.uint8)
        #new_mask
        return new_mask
 
    def inference_remove(self, inputs):
        image_dir, to_be_removed_txt = inputs.split(",")[0], ','.join(inputs.split(',')[1:])
        return self.inference_replace_sam(f"{image_dir},{to_be_removed_txt},background")

    def inference_replace_sam(self,inputs):
        image_input, to_be_replaced_txt, replace_with_txt = inputs.split(",")
        print(f"image_input={image_input}, to_be_replaced_txt={to_be_replaced_txt}")
        if os.path.isdir(image_input):
            image_paths = [
                os.path.join(image_input, path) for path in os.listdir(image_input)
            ]
        else:
            image_paths = image_input.split(",")
        updated_image_paths = []
        for image_path in image_paths:
            if "replace-something" in image_path:
                continue
            image_pil, image = self.grounding.load_image(image_path)
            boxes_filt, pred_phrases = self.grounding.get_grounding_boxes(image, to_be_replaced_txt)
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.sam.sam_predictor.set_image(image)
            masks = self.sam.get_mask_with_boxes(image_pil, image, boxes_filt)
            mask = torch.sum(masks, dim=0).unsqueeze(0)
            mask = torch.where(mask > 0, True, False)
            mask = mask.squeeze(0).squeeze(0).cpu() #tensor

            mask = self.pad_edge(mask,padding=20) #numpy
            mask_image = Image.fromarray(mask)

            updated_image = self.inpaint(prompt=replace_with_txt, image=image_pil,
                                        mask_image=mask_image)
            updated_image_path = get_new_image_name(image_path, func_name="replace-something")
            updated_image_paths.append(updated_image_path)
            updated_image = updated_image.resize(image_pil.size)
            updated_image.save(updated_image_path)
            print(
                f"\nProcessed ImageEditing, Input Image: {image_path}, Replace {to_be_replaced_txt} to {replace_with_txt}, "
                f"Output Image: {updated_image_path}")
        return updated_image_paths
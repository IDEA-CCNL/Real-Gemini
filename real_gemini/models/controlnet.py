import os
import random
from PIL import Image
import uuid

from controlnet_aux import OpenposeDetector

from .utils import get_new_image_name


class Image2Pose(object):
    def __init__(self, device="cuda"):
        self.detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')


    def __call__(self, image_input: str):
        if os.path.isdir(image_input):
            image_paths = [
                os.path.join(image_input, path) for path in os.listdir(image_input)
            ]
        else:
            image_paths = image_input.split(",")
        updated_image_paths = []
        for image_path in image_paths:
            if "human-pose" in image_path:
                continue
            image = Image.open(image_path)
            pose = self.detector(image)
            updated_image_path = get_new_image_name(image_path, func_name="human-pose")
            updated_image_paths.append(updated_image_path)
            pose.save(updated_image_path)
            print(f"\nProcessed Image2Pose, Input Image: {image_path}, Output Pose: {updated_image_path}")
        return updated_image_paths

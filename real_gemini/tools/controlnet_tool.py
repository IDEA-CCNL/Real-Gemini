import os
import random
from PIL import Image
import uuid

from controlnet_aux import OpenposeDetector

from .utils import get_new_image_name


class Image2Pose(object):
    _name_="Pose Detection On Image"
    _description_="""当你想要检测图片中的人体姿态时很有用。比如：生成这张图片中的人体姿态，或者从这张图片中生成人体姿态。输入到这个工具的应该是一个字符串，表示图片的路径或者图片所在的文件夹路径。
                useful when you want to detect the human pose of the image. 
                like: generate human poses of this image, or generate a pose image from this image.
                The input to this tool should be a string, representing the image_dir or image_paths."""
    
    def __init__(self, device):
        print("Initializing Image2Pose")
        self.detector = OpenposeDetector.from_pretrained('lllyasviel/ControlNet')


    def inference(self, image_input: str):
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

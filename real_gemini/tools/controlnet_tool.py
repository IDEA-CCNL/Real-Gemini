import requests


class Image2PoseTool(object):
    _name_="Pose-Detection-On-Image"
    _description_="""当你想要检测图片中的人体姿态时很有用。比如：生成这张图片中的人体姿态，或者从这张图片中生成人体姿态。输入到这个工具的应该是一个字符串，表示图片的路径或者图片所在的文件夹路径。useful when you want to detect the human pose of the image. like: generate human poses of this image, or generate a pose image from this image. The input to this tool should be a string, representing the image_dir or image_paths."""
    _return_direct_ = False
    
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 6679


    def inference(self, image_input: str):
        url = f"http://{self.host}:{self.port}/image_to_pose"
        data = {"image_input": image_input}
        pose_response = requests.post(url, data=data)
        return pose_response.text

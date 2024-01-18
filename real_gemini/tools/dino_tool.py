import requests

class Text2BoxTool(object):
    _name_="Detect the Give Object"
    _description_="""当你只想检测或者找出图片中的某个物体时很有用。输入到这个工具的应该是一个被逗号分隔成两部分的字符串，分别表示图片的路径或者图片所在的文件夹路径和要检测的物体。useful when you only want to detect or find out given objects in the picture. The input to this tool should be a comma separated string of two, representing the image_dir or image_paths and the object to be detected, respectively."""
    _return_direct_ = False
    
    def __init__(self):
        self.host = "0.0.0.0"
        self.port = 6681
    
    def inference(self, inputs):
        url = f"http://{self.host}:{self.port}/text_to_box"
        data = {"inputs": inputs}
        box_response = requests.post(url, data=data)
        return box_response.text
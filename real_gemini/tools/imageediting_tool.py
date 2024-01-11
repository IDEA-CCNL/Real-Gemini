import requests

class ImageRemoveTool(object):
    _name_ = "Remove Something From The Image"
    _description_ = "当你想要从图片中移除某个物体或者某个物体的某个部分时很有用。输入到这个工具的应该是一个被逗号分隔成两部分的字符串，分别表示图片的路径或者图片所在的文件夹路径和要移除的物体。\nuseful when you want to remove the object or something from the photofrom its description or location.\nThe input to this tool should be a comma separated string of two, representing the image_dir or image_paths and the object need to be removed."
    _return_direct_ = False

    def __init__(self):
        self.host = "0.0.0.0"
        # self.host = "localhost"
        self.port = 6682
    
    def inference(self, inputs):
        url = f"http://{self.host}:{self.port}/image_remove"
        data = {"input": inputs}
        update_response = requests.post(url, data=data)
        return update_response.text

class ImageReplaceTool(object):
    _name_="Replace Something From The Image"
    _description_="当你想要用另一个物体替换图片中的某个物体或者某个物体的某个部分时很有用。输入到这个工具的应该是一个被逗号分隔成三部份的字符串，分别表示图片的路径或者图片所在的文件夹路径，要替换的物体以及要替换成的物体。\nuseful when you want to replace an object from the object description or location with another object from its description.\nThe input to this tool should be a comma separated string of three, representing the image_dir or image_paths, the object to be replaced, the object to be replaced with."
    _return_direct_ = False

    def __init__(self):
        self.host = "0.0.0.0"
        # self.host = "localhost"
        self.port = 6682

    def inference(self, inputs):
        url = f"http://{self.host}:{self.port}/image_replace"
        data = {"input": inputs}
        update_response = requests.post(url, data=data)
        return update_response.text
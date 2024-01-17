import requests

class SegmentingTool(object):
    _name_="Segment the Image"
    _description_="""当你想要对图片中的所有物体进行分割，但是不想对某个物体进行分割时很有用。比如：分割这张图片中的所有物体，或者在这张图片上生成分割结果，或者对这张图片进行分割，或者分割这张图片中的所有物体。输入到这个工具的应该是一个字符串，表示图片的路径或者图片所在的文件夹路径。
                useful when you want to segment all the part of the image, but not segment a certain object.like: segment all the object in this image, or generate segmentations on this image,
                or segment the image,"
                or perform segmentation on this image, "
                or segment all the object in this image."
                The input to this tool should be a string, representing the image_dir or image_paths."""
    _return_direct_ = False
    
    def __init__(self):
        # self.host = "localhost"
        self.host = "0.0.0.0"
        self.port = 6680

    def inference(self,image_input: str):
        url = f"http://{self.host}:{self.port}/segmenting"
        data = {"image_input": image_input}
        seg_response = requests.post(url, data=data)
        return seg_response.text
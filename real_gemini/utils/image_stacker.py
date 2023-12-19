from PIL import Image
from io import BytesIO
import io
import base64

# 加载本地图像的示例函数
def load_images(paths):
    images = []
    for path in paths:
        image = Image.open(path)
        images.append(image)
    return images

def load_image(path):
    image = Image.open(path)
    return image

# 保存或展示处理后图像的函数
def save_or_show_image(base64_data, save_path=None):
    image_data = base64.b64decode(base64_data)
    image = Image.open(io.BytesIO(image_data))
    
    if save_path:
        image.save(save_path)
    else:
        image.show()

# 图像处理函数
def scale_and_stack_images(images, scale_factor=0.3):
    if not images:
        return None

    width, height = images[0].size
    scaled_width, scaled_height = int(width * scale_factor), int(height * scale_factor)
    canvas = Image.new('RGB', (scaled_width, scaled_height * len(images)))

    for index, image in enumerate(images):
        scaled_image = image.resize((scaled_width, scaled_height))
        canvas.paste(scaled_image, (0, index * scaled_height))

    buffered = BytesIO()
    canvas.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()

def image2base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def base642image(base64_data):
    image_data = base64.b64decode(base64_data)
    return Image.open(io.BytesIO(image_data))

if __name__=="__main__":
    image_paths = ["./test/images/test_0.png", "./test/images/test_1.png", "./test/images/test_2.png", "./test/images/test_3.png"]
    images = load_images(image_paths)
    print(image2base64(images[0]))
    stacked_image_base64 = scale_and_stack_images(images)
    # 保存或展示处理后的图像
    save_or_show_image(stacked_image_base64, "./test/outputs/stacked_image.jpg") # 保存到文件
    # 或者直接展示图像
    # save_or_show_image(stacked_image_base64)
import cv2
from datetime import datetime
import threading
import time
import numpy as np
import streamlit
 
class Camera(object):
 
    def __init__(self, video_path):
        #如果是笔记本调用外界摄像头，则把0改为1
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.ret, self.frame = self.cap.read()
        FPS = 24.0
        # 视频写入的图像尺寸与画布尺寸不对应会导致视频无法播放，需要实时获取
        WIDTH = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        HEIGHT = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # 设置摄像头设备分辨率
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
        # 设置摄像头设备帧率,如不指定,默认600
        self.cap.set(cv2.CAP_PROP_FPS, 24)
        # 建议使用XVID编码,图像质量和文件大小比较都兼顾的方案
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.out = cv2.VideoWriter(video_path, fourcc, FPS, (WIDTH, HEIGHT))
 
    def picture_shoot(self, image_name, image_path=None) -> None:
        '''
        调用摄像头拍照并保存图片到本地
        :param image_name: 图片名
        :param image_path: 图片保存路径
        :return: None
        '''
        self.image_name = image_name
        self.image_path = image_path
        cv2.imwrite(f'./{self.image_name}', self.frame)
 
    def video_record(self, video_path) -> None:
        '''
        调用摄像头录制视频并保存到本地
        :param video_path: 视频保存路径
        :return: None
        '''
        print('kaishiluzhi????')
        self.video_path = video_path
        while (self.cap.isOpened()):
            self.ret, self.frame = self.cap.read()
            if self.ret:
                font = cv2.FONT_HERSHEY_SIMPLEX
                datet = str(datetime.now())
                frame = cv2.putText(self.frame, datet, (10, 50), font, 1,
                                    (0, 255, 255), 2, cv2.LINE_AA)
                self.out.write(frame)
 
    def video_stop(self):
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        print('视频录制结束')
 
 
def record_v_a():

    # video_name = datetime.now().strftime("%Y%m%d%H%M%S") + ".mp4"
    width = 640
    height = 480

    # fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    # video_writer = cv2.VideoWriter(video_name, fourcc, 25, (width, height))

    capture = cv2.VideoCapture(0)

    while(True):
        ret, frame = capture.read()
        if ret:
            frame = cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)

            video_writer.write(frame)
            st.image(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    video_writer.release()
    capture.release()
    cv2.destroyAllWindows()

# if __name__ == '__main__':
#     # 录像+拍照
#     path = r'./video.mp4'
#     camera = Camera(path)
#     thread = threading.Thread(target=camera.video_record, args=(path,))
#     thread.start()
#     # camera.picture_shoot(image_name='1.png', image_path='./')
#     time.sleep(10)
#     camera.video_stop()
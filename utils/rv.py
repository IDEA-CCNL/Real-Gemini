
import numpy as np
import cv2  # OpenCV is used only for writing text on image (for testing).
import av
import io
import streamlit as st
n_frmaes = 100  # Select number of frames (for testing).
width, height, fps = 192, 108, 10  # Select video resolution and framerate.
output_memory_file = io.BytesIO()  # Create BytesIO "in memory file".
output = av.open(output_memory_file, 'w', format="mp4")  # Open "in memory file" as MP4 video output
stream = output.add_stream('h264', str(fps))  # Add H.264 video stream to the MP4 container, with framerate = fps.
stream.width = width  # Set frame width
stream.height = height  # Set frame height
#stream.pix_fmt = 'yuv444p'   # Select yuv444p pixel format (better quality than default yuv420p).
stream.pix_fmt = 'yuv420p'   # Select yuv420p pixel format for wider compatibility.
stream.options = {'crf': '17'}  # Select low crf for high quality (the price is larger file size).
def make_sample_image(i):
    """ Build synthetic "raw BGR" image for testing """
    p = width//60
    img = np.full((height, width, 3), 60, np.uint8)
    cv2.putText(img, str(i+1), (width//2-p*10*len(str(i+1)), height//2+p*10), cv2.FONT_HERSHEY_DUPLEX, p, (255, 30, 30), p*2)  # Blue number
    return img
# Iterate the created images, encode and write to MP4 memory file.
for i in range(n_frmaes):
    img = make_sample_image(i)  # Create OpenCV image for testing (resolution 192x108, pixel format BGR).
    frame = av.VideoFrame.from_ndarray(img, format='bgr24')  # Convert image from NumPy Array to frame.
    packet = stream.encode(frame)  # Encode video frame
    output.mux(packet)  # "Mux" the encoded frame (add the encoded frame to MP4 file).
# Flush the encoder
packet = stream.encode(None)
output.mux(packet)
output.close()
output_memory_file.seek(0)  # Seek to the beginning of the BytesIO.
#video_bytes = output_memory_file.read()  # Convert BytesIO to bytes array
#st.video(video_bytes)
st.video(output_memory_file)  # Streamlit supports BytesIO object - we don't have to convert it to bytes array.
# Write BytesIO from RAM to file, for testing:
#with open("output.mp4", "wb") as f:
#    f.write(output_memory_file.getbuffer())
#video_file = open('output.mp4', 'rb')
#video_bytes = video_file.read()
#st.video(video_bytes)
img = st.camera_input('test',label_visibility='hidden')
if img:
    st.image(img)
import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image

def histogram_equalization(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)

def white_balance(img):
    result = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    avg_a = np.average(result[:, :, 1])
    avg_b = np.average(result[:, :, 2])
    result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
    result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)

def enhance_image(img):
    img = histogram_equalization(img)
    img = white_balance(img)
    return img

def process_video_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    enhanced_frame = enhance_image(frame_rgb)
    return cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)

# Streamlit UI
st.title("Underwater Media Enhancer")

uploaded_file = st.file_uploader("Upload Image or Video", type=["png", "jpg", "jpeg", "mp4", "avi", "mov"])
if uploaded_file is not None:
    file_ext = uploaded_file.name.split('.')[-1].lower()

    if file_ext in ["png", "jpg", "jpeg"]:
        image = Image.open(uploaded_file)
        enhanced_image = enhance_image(np.array(image))
        st.image(enhanced_image, caption='Enhanced Image', use_column_width=True)

    elif file_ext in ["mp4", "avi", "mov"]:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_ext}') as tmpfile:
            tmpfile.write(uploaded_file.read())
            video_path = tmpfile.name

        st.text("Processing the video... Please wait.")
        output_path = video_path.split('.')[0] + '_enhanced.mp4'
        
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            enhanced_frame = process_video_frame(frame)
            out.write(enhanced_frame)

        cap.release()
        out.release()

        st.text("Processing complete!")

        with open(output_path, "rb") as file:
            btn = st.download_button(
                label="Download Enhanced Video",
                data=file,
                file_name="enhanced_video.mp4",
                mime="video/mp4"
            )

        os.remove(video_path)
        os.remove(output_path)
    else:
        st.error("Unsupported file type.")

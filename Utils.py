import numpy as np
import cv2
from skimage.color import rgb2hsv, hsv2rgb

def global_stretching(img_channel, height, width):
    I_min = np.min(img_channel)
    I_max = np.max(img_channel)

    stretched_channel = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            stretched_channel[i, j] = (img_channel[i, j] - I_min) * (1 / (I_max - I_min))
    
    return stretched_channel

def stretching(img):
    height, width = img.shape[:2]
    for k in range(3):
        Max_channel = np.max(img[:, :, k])
        Min_channel = np.min(img[:, :, k])
        img[:, :, k] = (img[:, :, k] - Min_channel) * (255 / (Max_channel - Min_channel))

    return img

def HSVStretching(sceneRadiance):
    height, width = sceneRadiance.shape[:2]
    img_hsv = rgb2hsv(sceneRadiance)
    h, s, v = cv2.split(img_hsv)

    img_s_stretching = global_stretching(s, height, width)
    img_v_stretching = global_stretching(v, height, width)

    hsv_stretched = np.zeros((height, width, 3), 'float64')
    hsv_stretched[:, :, 0] = h
    hsv_stretched[:, :, 1] = img_s_stretching
    hsv_stretched[:, :, 2] = img_v_stretching

    return hsv2rgb(hsv_stretched) * 255

def sceneRadianceRGB(sceneRadiance):
    sceneRadiance = np.clip(sceneRadiance, 0, 255)
    return np.uint8(sceneRadiance)

def RecoverICM(img):
    img = stretching(img)
    sceneRadiance = sceneRadianceRGB(img)
    sceneRadiance = HSVStretching(sceneRadiance)
    return sceneRadianceRGB(sceneRadiance)

def process_video_frame(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    enhanced_frame = RecoverICM(frame_rgb)
    return cv2.cvtColor(enhanced_frame, cv2.COLOR_RGB2BGR)
